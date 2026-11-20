"""Passive sampled-row selection and actual movement counters; no RNG or losses."""
import torch
class Usage:
    def __init__(self,table):
        self.draws=torch.zeros(len(table),device=table.device,dtype=torch.long)
        self.selected_steps=torch.zeros_like(self.draws);self.moved_steps=torch.zeros_like(self.draws)
        self.unsampled_moved_steps=torch.zeros_like(self.draws);self.steps=0
        self.unique_batch_histogram=[0]*33
    @torch.no_grad()
    def observe(self,table,before,indices):
        self.draws.index_add_(0,indices,torch.ones_like(indices,dtype=torch.long))
        rows=torch.unique(indices);selected=torch.zeros_like(self.draws,dtype=torch.bool);selected[rows]=True
        moved=(table.detach()!=before).any(-1)
        self.selected_steps+=selected.long();self.moved_steps+=moved.long()
        self.unsampled_moved_steps+=(moved&~selected).long();self.steps+=1
        self.unique_batch_histogram[len(rows)]+=1
    def state_dict(self):
        return dict(steps=self.steps,draws=self.draws.cpu().clone(),selected_steps=self.selected_steps.cpu().clone(),
          moved_steps=self.moved_steps.cpu().clone(),unsampled_moved_steps=self.unsampled_moved_steps.cpu().clone(),
          unique_batch_histogram=list(self.unique_batch_histogram))
