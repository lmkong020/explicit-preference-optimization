import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import random

from tqdm import tqdm

class Model(nn.Module):
    def __init__(
        self, 
        theta, 
        loss_type='dpo',
        beta=0.,
        alpha=None,
        pi_star_probs=None
    ) -> None:
        super().__init__()
        assert loss_type in ['dpo', 'ipo', 'f-dpo', 'expo', 'rlhf']
        self.loss_type = loss_type
        self.beta = beta
        self.alpha = alpha
        if pi_star_probs is not None:
            self.pi_star_logprobs = torch.log(torch.tensor(pi_star_probs, device='cuda'))
        
        self.policy = torch.tensor(theta, requires_grad=True, device='cuda')
        self.reference = torch.tensor(theta, requires_grad=False, device='cuda')

    def forward(self, y_w, y_l, x=None):
        if x is not None:
            policy_logits = torch.tensor(x).matmul(self.policy).squeeze()
            reference_logits = torch.tensor(x).matmul(self.reference).squeeze()
        else:
            policy_logits = self.policy
            reference_logits = self.reference
        
        pi_logps = F.log_softmax(policy_logits, -1)
        ref_logps = F.log_softmax(reference_logits, -1)
        pi_logratios = pi_logps[y_w] - pi_logps[y_l]
        ref_logratios = ref_logps[y_w] - ref_logps[y_l]
        logits = pi_logratios - ref_logratios
        if self.loss_type == 'dpo':
            loss = -F.logsigmoid(self.beta * logits)
        elif self.loss_type == 'ipo':
            loss = (logits - 1/(2 * self.beta)) ** 2
        elif self.loss_type == 'expo':
            sup_loss = torch.log1p(torch.exp(-pi_logratios))
            unsup_loss = F.kl_div(pi_logps, ref_logps, log_target=True)
            loss = sup_loss + self.beta * unsup_loss
        elif self.loss_type == 'f-dpo':
            pi_probs = F.softmax(policy_logits, -1)
            ref_probs = F.softmax(reference_logits, -1)
            t1 = torch.log(2*pi_probs[y_w]) - torch.log(pi_probs[y_w] + ref_probs[y_w])
            t2 = torch.log(2*pi_probs[y_l]) - torch.log(pi_probs[y_l] + ref_probs[y_l])
            loss = -F.logsigmoid(self.beta * (t1 - t2))
        elif self.loss_type == 'rlhf':
            # r_w = self.pi_star_logprobs[y_w] - ref_logps[y_w] + self.beta * (pi_logps[y_w] - ref_logps[y_w])
            # r_l = self.pi_star_logprobs[y_l] - ref_logps[y_l] + self.beta * (pi_logps[y_l] - ref_logps[y_l])
            # loss = (-pi_logps[y_w] * r_w + -pi_logps[y_l] * r_l) / 2.
            
            # r_w = self.pi_star_logprobs[y_w] - self.beta * (pi_logps[y_w] - ref_logps[y_w])
            # r_l = self.pi_star_logprobs[y_l] - self.beta * (pi_logps[y_l] - ref_logps[y_l])
            # loss = (-pi_logps[y_w] * r_w + -pi_logps[y_l] * r_l) / 2.

            reward = self.pi_star_logprobs - self.beta * (pi_logps - ref_logps)
            loss = (- torch.exp(pi_logps) * reward).sum()
            
        if self.alpha is not None:
            loss += self.alpha * ((self.policy ** 2).sum())
        return loss.mean()


def recover_params_with_probs(probs):
    theta = torch.randn(3, requires_grad=True)
    learning_rate = 0.1
    num_iterations = 3000
    
    for _ in range(num_iterations):
        pred_probs = F.softmax(theta, -1)
        loss = torch.mean((pred_probs - probs) ** 2)
        loss.backward()
        with torch.no_grad():
            theta -= learning_rate*theta.grad
        theta.grad.zero_()
    return theta.data.cpu().numpy().tolist()


def generate_data(probs, n):
    p_star = np.zeros([3, 3])
    for i in range(3):
        for j in range(3):
            p_star[i][j] = probs[i] / (probs[i] + probs[j])

    data = []
    for _ in range(n):
        y1 = random.sample([0, 1, 2], k=1)[0]
        y2 = random.sample([0, 1, 2], k=1)[0]
        if y1 != y2:
            if random.random() < p_star[y1][y2]:
                data.append((y1, y2))
            else:
                data.append((y2, y1))
    return data


def train(data, init_theta, loss_type, beta, num_epochs, alpha=None, pi_star_probs=None, lr=5e-4, batch_size=100):
    print(f'Training {loss_type}')
    model = Model(init_theta, loss_type, beta, alpha=alpha, pi_star_probs=pi_star_probs).cuda()
    optimizer = torch.optim.Adam(params=[model.policy], lr=lr)
    # optimizer = torch.optim.SGD(params=[model.policy], lr=lr)
    max_grad_norm = 10.0
    
    step = 0
    
    loss_list = []
    
    step_list = []
    probs_list = []
    for epoch in range(num_epochs):
        for i in range(0, len(data), batch_size):
            y_w = torch.tensor([v[0] for v in data[i: i+batch_size]]).cuda()
            y_l = torch.tensor([v[1] for v in data[i: i+batch_size]]).cuda()
            if data[0][2] is not None:
                x = [v[2] for v in data[i: i+batch_size]]
                x = torch.stack(x, 0).cuda()
            else:
                x = None    
            
            loss = model(y_w, y_l, x)
            loss.backward()
            nn.utils.clip_grad_norm_([model.policy], max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

            step += 1
            
            if step % 50 == 0:
                loss_list.append(loss.item())
                step_list.append(step)
                with torch.no_grad():
                    probs_list.append(F.softmax(model.policy, -1).data.cpu().numpy().tolist())
    return model.policy.data.detach(), step_list, probs_list, loss_list