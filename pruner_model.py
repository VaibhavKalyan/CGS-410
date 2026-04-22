import math
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel

def _get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class HardConcreteHeadMask(nn.Module):
    def __init__(self, num_layers, num_heads, init_mean=0.5, temp=0.1, stretch=0.1):
        super().__init__()
        self.temp, self.stretch = temp, stretch
        self.l, self.r = -stretch, 1.0 + stretch
        self.log_alpha = nn.Parameter(torch.empty(num_layers, num_heads).normal_(init_mean, 0.01))

    def forward(self, training=True):
        if training:
            u = torch.rand_like(self.log_alpha).clamp(1e-8, 1.0 - 1e-8)
            s = torch.sigmoid((torch.log(u) - torch.log(1.0 - u) + self.log_alpha) / self.temp)
            s = s * (self.r - self.l) + self.l
            return torch.clamp(s, 0.0, 1.0)
        else:
            mask = torch.sigmoid(self.log_alpha) * (self.r - self.l) + self.l
            return torch.clamp(mask, 0.0, 1.0)

    def get_l0_penalty(self):
        return torch.sigmoid(self.log_alpha - self.temp * math.log(-self.l / self.r)).sum()

    def get_active_heads(self, threshold=0.5):
        return (self.forward(training=False) > threshold).sum().item()

class GPT2PrunerWrapper(nn.Module):
    def __init__(self, model_name='gpt2-medium'):
        super().__init__()
        self.device = _get_device()
        self.model = GPT2LMHeadModel.from_pretrained(model_name, attn_implementation='eager').to(self.device)
        self.config = self.model.config
        self.num_layers, self.num_heads = self.config.n_layer, self.config.n_head
        self.head_mask_module = HardConcreteHeadMask(self.num_layers, self.num_heads).to(self.device)

    def forward(self, input_ids, attention_mask=None, head_mask=None, output_attentions=True, output_hidden_states=True):
        input_ids = input_ids.to(self.device)
        if attention_mask is not None: attention_mask = attention_mask.to(self.device)
        if head_mask is None: head_mask = self.head_mask_module(training=self.training)
        
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask, head_mask=head_mask,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=True
        ), head_mask

def train_l0_pruning(model, inputs, target_active, epochs=50, lr=0.02, lambda_l0=1.0, batch_size=8, verbose=True):
    device = model.device
    for p in model.model.parameters(): p.requires_grad_(False)
    model.head_mask_module.log_alpha.requires_grad_(True)
    optimizer = torch.optim.Adam([model.head_mask_module.log_alpha], lr=lr)

    n_samples = inputs['input_ids'].size(0)
    model.train()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        indices = torch.randperm(n_samples)
        
        for i in range(0, n_samples, batch_size):
            b_idx = indices[i:i+batch_size]
            b_in = {k: v[b_idx].to(device) for k, v in inputs.items()}
            
            outputs, _ = model(b_in['input_ids'], attention_mask=b_in['attention_mask'], output_attentions=False, output_hidden_states=False)
            
            logits = outputs.logits
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = b_in['input_ids'][:, 1:].contiguous()
            lm_loss = nn.CrossEntropyLoss()(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            # Scale loss by batch proportion so L0 penalty balances correctly
            lm_loss = lm_loss * (len(b_idx) / n_samples)
            
            l0_penalty = model.head_mask_module.get_l0_penalty()
            l0_loss = lambda_l0 * (l0_penalty - target_active).abs() * (len(b_idx) / n_samples)
            
            (lm_loss + l0_loss).backward()
            
        optimizer.step()
        
    for p in model.model.parameters(): p.requires_grad_(True)
    model.eval()

def compute_integrated_gradients(model, inputs, labels, dep_metadata, n_steps=20, batch_size=8):
    model.eval()
    device = model.device
    num_layers, num_heads = model.num_layers, model.num_heads
    n_samples = inputs['input_ids'].size(0)

    baseline = torch.zeros(num_layers, num_heads, device=device)
    target = model.head_mask_module.forward(training=False).detach().clone()
    attributions = torch.zeros(num_layers, num_heads, device=device)

    for step in range(n_steps + 1):
        alpha = step / n_steps
        interpolated = baseline + alpha * (target - baseline)

        r_val, l_val = model.head_mask_module.r, model.head_mask_module.l
        sig_val = ((interpolated - l_val) / (r_val - l_val)).clamp(1e-6, 1.0 - 1e-6)
        
        fake_log_alpha = torch.log(sig_val / (1.0 - sig_val)).detach().requires_grad_(True)
        mask = (torch.sigmoid(fake_log_alpha) * (r_val - l_val) + l_val).clamp(0.0, 1.0)

        # Batch the IG calculation
        for i in range(0, n_samples, batch_size):
            b_in = {k: v[i:i+batch_size].to(device) for k, v in inputs.items()}
            outputs = model.model(
                input_ids=b_in['input_ids'], attention_mask=b_in['attention_mask'],
                head_mask=mask, output_attentions=False, output_hidden_states=False, return_dict=True
            )
            shift_logits = outputs.logits[:, :-1, :].contiguous()
            shift_labels = b_in['input_ids'][:, 1:].contiguous()
            
            loss = nn.CrossEntropyLoss()(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = loss * (len(b_in['input_ids']) / n_samples)
            loss.backward()

        if fake_log_alpha.grad is not None:
            attributions += fake_log_alpha.grad.detach()

    attributions = (target - baseline) * attributions / (n_steps + 1)
    return attributions.abs()

def get_expert_heads(attributions, top_k=10):
    flat = attributions.view(-1)
    k = min(top_k, flat.numel())
    _, topk_idx = torch.topk(flat, k)
    return [(idx.item() // attributions.size(1), idx.item() % attributions.size(1)) for idx in topk_idx]
