import openai
import torch as tc
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer

from addict import Dict

MODELS = {
    'qwen3-0.6b': 'Qwen/Qwen3-0.6B',
    'qwen3-4b': 'Qwen/Qwen3-8B',
    'qwen3-8b': 'Qwen/Qwen3-4B',
    'qwen3-14b': 'Qwen/Qwen3-14B',
    'qwen3-32b': 'Qwen/Qwen3-32B',
    'llama3-8b': 'meta-llama/Llama-3.1-8B-Instruct',
    'llama4-mav': 'meta-llama/Llama-4-Maverick-17B-128E-Instruct',
    'deepseek-v3': 'deepseek-ai/DeepSeek-V3',
    'hunyuan-a13b': 'tencent/Hunyuan-A13B-Instruct',
    'gemma3-27b': 'google/gemma-3-27b-it'
}

MODEL_NAMES = sorted(list(MODELS.keys()))

CACHED_MODELS = {}

def get_model(model_name, device='cuda'):
    model_name = model_name.lower()
    model_name = MODELS[model_name]
    
    if model_name in CACHED_MODELS:
        model = CACHED_MODELS[model_name]
    else:
        # init model
        model = HFTransformer(model_name, device)
        CACHED_MODELS[model_name] = model

    return model

class LLM:

    def __init__(self, model_name):
        self.model_name = model_name

    def format_messages(self, user_prompt, syst_prompt=None, msg_history=None):
        def message_format(role, content):
            if 'google/gemma-3' in self.model_name:
                return {
                    'role': role,
                    'content': [{
                        'type': 'text',
                        'text': content
                    },]
                }
            else:
                return {
                    'role': role,
                    'content': content
                }
        
        messages = []
        if syst_prompt:
            message = message_format('system', syst_prompt)
            messages.append(message)
            
        if msg_history: messages += msg_history

        message = message_format('user', user_prompt)
        messages.append(message)

        return messages
        
class HFTransformer(LLM):

    def __init__(self, model_name, device='cuda'):
        super().__init__(model_name=model_name)
        self.device = device
        
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map=device,
            trust_remote_code=True
        )
        self.model.eval()
        
        self.streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        self.eos_token = tc.tensor([self.tokenizer.eos_token_id]).to(device)
        
    def format_messages(self,
                        user_prompt,
                        syst_prompt=None,
                        msg_history=None,
                        add_generation_prompt=True,
                        thinking=False):

        messages = super().format_messages(user_prompt, syst_prompt, msg_history)
        messages = self.tokenizer.apply_chat_template(messages,
                                                      tokenize=False,
                                                      add_generation_prompt=add_generation_prompt,
                                                      enable_thinking=thinking)
        return messages
        
    def tokenize(self, text):
        tokenized = self.tokenizer([text], return_tensors="pt").to(self.device)
        return tokenized['input_ids'], tokenized['attention_mask']

    def decode(self, tokens, skip_special_tokens=True):
        tokens = tokens.tolist() if isinstance(tokens, tc.Tensor) else tokens
        return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

    def generation_configs(self, **args):
        args = Dict(args)
        if 'temperature' in args and args.temperature == 0.0:
            args.do_sample = False

        default = self.model.generation_config

        if not default.pad_token_id:
            args.pad_token_id = self.tokenizer.eos_token_id
            
        if args.do_sample != {} and not args.do_sample:
            args.temperature = 1.0
            args.top_p = None
            args.top_k = None
            
        return args
            
    def next_token(self, tokens, attention_mask=None, do_sample=True, temperature=0.6, top_k=50, top_p=1):

        args = self.generation_configs(do_sample=do_sample,
                                       temperature=temperature,
                                       top_k=top_k,
                                       top_p=top_p)
        all_tokens = self.model.generate(input_ids=tokens,
                                         max_new_tokens=1,
                                         attention_mask=attention_mask,
                                         **args)
        next_token = all_tokens[0][-1:]
        
        # outputs = self.model(input_ids=tokens, attention_mask=attention_mask)
        # next_token_logits = outputs.logits[:, -1, :]
        # # greedy decode
        # next_token = tc.argmax(next_token_logits, dim=-1)
        
        return next_token
    
    def complete(self,
                 user_prompt,
                 syst_prompt=None,
                 msg_history=None,
                 max_new_tokens=10000,
                 do_sample=True,
                 temperature=0.6,
                 top_k=50,
                 top_p=1,
                 repetition_penalty=1.0):

        messages = self.format_messages(user_prompt=user_prompt,
                                        syst_prompt=syst_prompt,
                                        msg_history=msg_history)
        
        input_tokens, attention_mask = self.tokenize(messages)

        args = self.generation_configs(do_sample=do_sample,
                                       temperature=temperature,
                                       top_k=top_k,
                                       top_p=top_p)
        
        all_tokens = self.model.generate(input_ids=input_tokens,
                                         max_new_tokens=max_new_tokens,
                                         attention_mask=attention_mask,
                                         streamer=self.streamer,
                                         repetition_penalty=repetition_penalty,
                                         eos_token_id=self.eos_token,
                                         **args)
        gen_tokens = all_tokens[0][input_tokens.size(1):]

        return self.decode(gen_tokens)


class APILLM(LLM):
    
    def __init__(self, model_name, client):
        super().__init__(model_name)

        self.client = client

    def complete(self,
                 user_prompt,
                 syst_prompt=None,
                 msg_history=None,
                 temperature=None,
                 max_new_tokens=10000):
        messages = self.format_messages(user_prompt, syst_prompt, msg_history)
        response = self._complete(messages,
                                  temperature=temperature,
                                  max_new_tokens=max_new_tokens)
        return response

class OpenAILLM(APILLM):
    
    def __init__(self, model_name, client):
        super().__init__(model_name, client)
        
    def _complete(self, messages, temperature, max_new_tokens):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_completion_tokens=max_new_tokens
        )
        response = response.choices[0].message.content
        return response
        
class GPT(OpenAILLM):

    def __init__(self, model_name):
        client = openai.OpenAI()
        super().__init__(model_name, client)

class DeepSeek(OpenAILLM):

    def __init__(self, model_name):
        client = openai.OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"],
                               base_url="https://api.deepseek.com")
        super().__init__(model_name, client)
        

    
if __name__ == '__main__':
    model = get_model('qwen3-32B')

    # print(model.generation_config)
    
    # print(model.complete('Who is Alan Turing?'))

    # input_tokens, attention_mask = model.tokenize('John Russell Reynolds was a prominent British physician and one of the founding figures of modern clinical medicine in the United Kingdom.')
    # all_tokens = model.model.generate(input_ids=input_tokens,
    #                                   max_new_tokens=1000,
    #                                   attention_mask=attention_mask)
    # gen_tokens = all_tokens[0][input_tokens.size(1):]
    # print(model.decode(gen_tokens))
