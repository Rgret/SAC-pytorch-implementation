import numpy as np
import atexit
import os.path
import time
import utils

# based on https://github.com/openai/spinningup/blob/master/spinup/utils/logx.py

class Logger():
    def __init__(self, output_dir = None, output_fname="logs.txt", exp_name=None):
        self.output_dir = output_dir or "/Network/tests/%i"%int(time.time())
        if(os.path.exists(self.output_dir)):
            print("Log file exists")
        else:
            os.makedirs(self.output_dir)
        self.output_file = open(os.path.join(self.output_dir, output_fname), 'w')
        atexit.register(self.output_file.close)
        self.first_row = True
        self.log_headers = []
        self.log_current_row = {}
        self.exp_name = exp_name
        self.epoch_dict = dict()

    def store(self, **kwargs):
        for k, v in kwargs.items():
            if not (k in self.epoch_dict.keys()):
                self.epoch_dict[k] = []
            self.epoch_dict[k].append(v)
    
    def log_tabular_(self, key, val, with_min_and_max=False, avg_only=False):
        if self.first_row:
            self.log_headers.append(key)
        else:
            assert key in self.log_headers, "New key"
        assert key not in self.log_current_row, "Call dump_tabular before setting same key again"
        self.log_current_row[key] = val

    def log_tabular(self, key, val=None, with_min_and_max=False, avg_only=False):
        if val is not None:
            self.log_tabular_(key, val)
        else:
            v = self.epoch_dict[key]
            vals = np.concatenate(v) if isinstance(v[0], np.ndarray) and len(v[0].shape)>0 else v
            stats = utils.get_stats(vals, min_max = with_min_and_max)
            self.log_tabular_(key if avg_only else 'Average' + key, stats[0])
            if not(avg_only):
                self.log_tabular_('Std'+key, stats[1])
            if with_min_and_max:
                self.log_tabular_('Min'+key, stats[2])
                self.log_tabular_('Max'+key, stats[3])
        self.epoch_dict[key] = []

    def dump_tabular(self):
        vals = []
        key_lens = [len(key) for key in self.log_headers]
        max_key_len = max(15,max(key_lens))
        keystr = '%'+'%d'%max_key_len
        fmt = "| " + keystr + "s | %15s |"
        n_slashes = 22 + max_key_len
        print("-"*n_slashes)
        for key in self.log_headers:
            val = self.log_current_row.get(key, "")
            valstr = "%8.3g"%val if hasattr(val, "__float__") else val
            print(fmt%(key, valstr))
            vals.append(val)
        print("-"*n_slashes, flush=True)
        if self.output_file is not None:
            if self.first_row:
                self.output_file.write("\t".join(self.log_headers)+"\n")
            self.output_file.write("\t".join(map(str,vals))+"\n")
            self.output_file.flush()
        self.log_current_row.clear()
        self.first_row=False
        