# Gadget-assisted Language Models

This repo contains dataset builders, training scripts 
and inference wrappers for training and using 
Tool-assisted Language Models. 

Instructions below cover the **usage** of our tool-supported models 
and **reproduction** of our results, models and datasets reported in the paper.

## Usage

We wrap the `generate()` method to be able to utilize the 
given set of gadgets during the generation. 
You will need to wrap the model of your choice and 
make sure that the tokenizer is able to encode the instruction
HTML tags used in calling the gadget calls.

Using our pre-trained models (with the tokenizer resolved),
you can use the model using calculator gadget as follows.

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer

from gadgets.gadget_assisted_model import GadgetAssistedModel
from gadgets.gadget import Calculator


class GadgetAssistedT5(GadgetAssistedModel, T5ForConditionalGeneration):
    # GadgetAssistedModel overrides the standard generate() from transformers
    pass


model = GadgetAssistedT5.from_pretrained("emnlp2023/calc-t5-large")
tokenizer = T5Tokenizer.from_pretrained("emnlp2023/calc-t5-large")

model.prepare_for_generate(tokenizer, 
                           enabled_gadgets=[Calculator()], 
                           default_max_tokens=512)
query = """
    The profit from a business transaction is shared among 2 business partners, 
    Mike and Johnson in the ratio 2:5 respectively. 
    If Johnson got $2500, how much will Mike have 
    after spending some of his share on a shirt that costs $200?
"""

inputs = tokenizer(query, return_tensors="pt")
output_ids = model.generate(**inputs)
tokenizer.decode(output_ids[0], spaces_between_special_tokens=False)

# This returns:
# 'According to the ratio, Mike got 2/5*$2500 = $<gadget id="calculator">2/5*2500</gadget><output>1_000</output> 1000 
#  Mike will have $1000-$200 = $<gadget id="calculator">1000-200</gadget><output>800</output> 800 after buying a shirt. 
#  Final result is<result>800</result></s>'
```

## Reproduction

### Evaluation

You can reproduce the results reported in the paper by running the following commands.

First, use some of the trained models to collect predictions: 

```shell
CUDA_VISIBLE_DEVICES=0 python examples/predict_calc.py --model_checkpoint emnlp2023/calc-flan-xl --dataset "emnlp2023/Calc-ape210k" --split "test" --output_jsonl predictions_calc_flan_3B.jsonl --use_gadgets
```

then, compute the accuracy of correct results with bootstrapped confidence intervals, as reported in the paper:

```shell
python examples/test_calc.py --input_jsonl predictions_calc_flan_3B.jsonl --use_gadgets
```


### Training

You can reproduce the training of (1) **baseline models** by running 
[examples/train_calc_balanced_baseline.py script](https://github.com/emnlp2023sub/gadgets/blob/65e24e810cf5ea20aceb8a3c8ddbc19f035ab694/examples/train_calc_balanced_baseline.py)
and (2) **calculator-supported model** by running [examples/train_calc_balanced.py script](https://github.com/emnlp2023sub/gadgets/blob/65e24e810cf5ea20aceb8a3c8ddbc19f035ab694/examples/train_calc_balanced.py) like this:

```shell
CUDA_VISIBLE_DEVICES=0 python examples/train_calc_balanced.py
```

All parameters of the training are intentionally fixed within the script. However, note that you might need to reconfigure some parameters affecting memory usage, depending on your hardware, such as batch_size, gradient_checkpointing and gradient_accumulation steps.

The training scripts also assume that you use Weights&Biases logging: reconfigure `wandb.init` by your settings, or simply remove the occurrences of `wandb` from the code if you do not wish to log.
