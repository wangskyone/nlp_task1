from torch.utils.data import DataLoader
from transformers.trainer import Trainer
from typing import Dict, List, Tuple, Optional, Any, Union
import torch
from torch import nn

class  CorrectTrainer(Trainer):
    def __init__(self, **kwds):
        super().__init__(**kwds)


    def compute_loss(self, model, inputs, return_outputs=False):
        src, trg  = inputs
        dic = model(input_ids=src, labels=trg, return_dict=True)
        return (dic['loss'], ) if return_outputs else dic['loss']
    
    def get_train_dataloader(self) -> DataLoader:
        train_dataloader = DataLoader(self.train_dataset, batch_size=self._train_batch_size, shuffle=True, pin_memory=True, num_workers=64)
        return self.accelerator.prepare(train_dataloader)

    def get_eval_dataloader(self, eval_dataset) -> DataLoader:
        eval_dataloader = DataLoader(self.eval_dataset, batch_size=self._train_batch_size, shuffle=False, pin_memory=True, num_workers=64)
        return self.accelerator.prepare(eval_dataloader)
    
    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        src, trg  = inputs

        # eval_loss = super().prediction_step(model, inputs, True, ignore_keys)[0]
        with torch.no_grad():
            # greedy search
            doc_ids = model.generate(
                src,
                max_length=100,
                early_stopping=True,)

        return (None, doc_ids, trg)