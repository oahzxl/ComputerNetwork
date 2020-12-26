import torch
import torch.nn as nn
from sim.utils import v_wrap, set_init, push_and_pull, record
import torch.nn.functional as f
import torch.multiprocessing as mp
from sim import load_trace, env
import os
import numpy as np

os.environ["OMP_NUM_THREADS"] = "1"

UPDATE_GLOBAL_ITER = 5
LR = 1e-4
GAMMA = 0.9
MAX_EP = 3000
N_S = 6  # bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
S_LEN = 8  # take how many frames in the past
N_A = 6
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
NUM_AGENTS = 16
TRAIN_SEQ_LEN = 100  # take as a train batch
MODEL_SAVE_INTERVAL = 100
VIDEO_BIT_RATE = [300, 750, 1200, 1850, 2850, 4300]  # Kbps
HD_REWARD = [1, 2, 3, 12, 15, 20]
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 48.0
M_IN_K = 1000.0
REBUF_PENALTY = 4.3  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1  # default video quality without agent
RANDOM_SEED = 42
RAND_RANGE = 1000
SUMMARY_DIR = './results'
LOG_FILE = './results/log'
TEST_LOG_FOLDER = './test_results/'
TRAIN_TRACES = './cooked_traces/'
# NN_MODEL = './results/pretrain_linear_reward.ckpt'
NN_MODEL = None


class SharedAdam(torch.optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8,
                 weight_decay=0):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        # State initialization
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

                # share in memory
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()


class Cat(torch.distributions.Categorical):
    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        value = value.long()
        value, log_pmf = torch.broadcast_tensors(value, self.logits)
        value = value[..., :1]
        return log_pmf.gather(-1, value).squeeze(-1)


class Net(nn.Module):
    def __init__(self, s_dim, a_dim, s_len):
        super(Net, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.s_len = s_len
        self.pi1 = nn.Linear(s_dim * s_len, 128)
        self.pi2 = nn.Linear(128, a_dim)

        self.p1 = nn.Sequential(
            nn.Conv1d(6, 16, kernel_size=2, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.p2 = nn.Sequential(
            nn.Linear(16 * 9, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )

        self.v1 = nn.Sequential(
            nn.Conv1d(6, 16, kernel_size=2, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.v2 = nn.Sequential(
            nn.Linear(16 * 9, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )

        # set_init([self.pi1, self.pi2, self.v1, self.v2])
        # self.distribution = torch.distributions.Categorical
        self.distribution = Cat

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        logits = self.p1(x)
        logits = logits.view(logits.size(0), -1)
        logits = self.p2(logits)

        values = self.v1(x)
        values = values.view(values.size(0), -1)
        values = self.p2(values)
        return logits, values

    def choose_action(self, s):
        self.eval()
        logits, _ = self.forward(s)
        prob = f.softmax(logits, dim=1).data
        # m = self.distribution(prob)
        # return m.sample().numpy()[0]
        return prob

    def loss_func(self, s, a, v_t):
        self.train()
        logits, values = self.forward(s)
        # v_t = v_t.squeeze(-1)
        # values = values.squeeze(-1)
        td = v_t.squeeze(-1).squeeze(-1) - values.squeeze(-1)
        c_loss = td.pow(2)

        probs = f.softmax(logits, dim=1)

        m = self.distribution(probs)
        exp_v = m.log_prob(a) * td.detach()
        a_loss = -exp_v
        total_loss = (c_loss + a_loss).mean()
        return total_loss


class Worker(mp.Process):
    def __init__(self, g_net_, opt_, global_ep_, global_ep_r_, res_queue_, name_):
        super(Worker, self).__init__()
        self.name = 'w%02i' % name_
        self.g_ep, self.g_ep_r, self.res_queue = global_ep_, global_ep_r_, res_queue_
        self.g_net, self.opt = g_net_, opt_
        self.l_net = Net(N_S, N_A, S_LEN)  # local network
        self.env = env.Environment(all_cooked_time=all_cooked_time,
                                   all_cooked_bw=all_cooked_bw,
                                   random_seed=name_)

    def run(self):
        total_step = 1
        while self.g_ep.value < MAX_EP:

            # state = self.env.reset()
            ep_r = 0.

            last_bit_rate = DEFAULT_QUALITY
            bit_rate = DEFAULT_QUALITY

            action_vec = torch.zeros(N_A)
            action_vec[bit_rate] = 1

            buffer_s = [torch.zeros((N_S, S_LEN))]
            buffer_a = [action_vec]
            buffer_r = []

            time_stamp = 0

            while True:

                delay, sleep_time, buffer_size, rebuf, \
                video_chunk_size, next_video_chunk_sizes, \
                end_of_video, video_chunk_remain = \
                    self.env.get_video_chunk(bit_rate)
                time_stamp += delay  # in ms
                time_stamp += sleep_time  # in ms

                reward = torch.tensor(VIDEO_BIT_RATE[bit_rate] / M_IN_K
                                      - REBUF_PENALTY * rebuf
                                      - SMOOTH_PENALTY * abs(VIDEO_BIT_RATE[bit_rate]
                                                             - VIDEO_BIT_RATE[last_bit_rate]) / M_IN_K)
                ep_r += reward
                buffer_r.append(reward)
                last_bit_rate = bit_rate

                # retrieve previous state
                if len(buffer_s) == 0:
                    state = torch.zeros((N_S, S_LEN))
                else:
                    state = buffer_s[-1].detach()

                # dequeue history record
                state = torch.roll(state, -1, dims=1)

                # this should be S_INFO number of terms
                state[0, -1] = VIDEO_BIT_RATE[bit_rate] / float(max(VIDEO_BIT_RATE))  # last quality
                state[1, -1] = buffer_size / BUFFER_NORM_FACTOR  # 10 sec
                state[2, -1] = float(video_chunk_size) / float(delay) / M_IN_K  # kilo byte / ms
                state[3, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
                state[4, :N_A] = torch.tensor(next_video_chunk_sizes) / M_IN_K / M_IN_K  # mega byte
                state[5, -1] = min(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)

                action_prob = self.l_net.choose_action(torch.reshape(state, (1, N_S, S_LEN)))
                # action_sum = torch.sum(action_prob)
                bit_rate = torch.argmax(action_prob *
                                        (action_prob > torch.tensor(np.random.randint(1, RAND_RANGE)
                                                                    / float(RAND_RANGE))))

                if end_of_video:
                    last_bit_rate = DEFAULT_QUALITY
                    bit_rate = DEFAULT_QUALITY  # use the default action here

                    action_vec = torch.zeros(N_A)
                    action_vec[bit_rate] = 1

                    buffer_s.append(torch.zeros((N_S, S_LEN)))
                    buffer_a.append(action_vec)

                else:
                    buffer_s.append(state)

                    action_vec = torch.zeros(N_A)
                    action_vec[bit_rate] = 1
                    buffer_a.append(action_vec)

                if total_step % UPDATE_GLOBAL_ITER == 0 or end_of_video:  # update global and assign to local net
                    # sync
                    push_and_pull(self.opt, self.l_net, self.g_net, end_of_video, state,
                                  buffer_s, buffer_a, buffer_r, GAMMA)
                    buffer_s, buffer_a, buffer_r = [torch.zeros((N_S, S_LEN))], [action_vec], []

                    if end_of_video:  # done and print information
                        record(self.g_ep, self.g_ep_r, ep_r, self.res_queue, self.name)
                        break
                total_step += 1
        self.res_queue.put(None)


if __name__ == "__main__":
    all_cooked_time, all_cooked_bw, _ = load_trace.load_trace(TRAIN_TRACES)

    global_net = Net(N_S, N_A, S_LEN)  # global network
    global_net.share_memory()  # share the global parameters in multiprocessing
    opt = SharedAdam(global_net.parameters(), lr=LR)  # global optimizer
    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()

    # parallel training
    workers = [Worker(global_net, opt, global_ep, global_ep_r, res_queue, i) for i in range(mp.cpu_count())]
    [w.start() for w in workers]
    res = []  # record episode reward to plot
    while True:
        r = res_queue.get()
        if r is not None:
            res.append(r)
        else:
            break
    [w.join() for w in workers]

    import matplotlib.pyplot as plt

    plt.plot(res)
    plt.ylabel('Moving average ep reward')
    plt.xlabel('Step')
    plt.show()
