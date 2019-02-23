```Python
import numpy as np
import pickle

from ClassActivateMap import ClassActivateMap
from visual import cam_visualization

class PreDataHelper:
    def __init__(self, char2id_dct_fp, max_sentence_len):
        self.UNK_IND = 0
        self.PAD_IND = 1
        self.BOS_IND = 2
        self.EOS_IND = 3

        self.max_sentence_len = max_sentence_len

        with open(char2id_dct_fp, 'rb') as f:
            self.char2id_dct = pickle.load(f)

    def char2id(self, sentence):
        sentence_inds = []
        for char in sentence.strip():
            sentence_inds.append(self.char2id_dct.get(char, self.UNK_IND))
        # add BOS and EOS
        sentence_inds = [self.BOS_IND] + sentence_inds + [self.EOS_IND]
        # add PAD
        true_length = len(sentence_inds)
        sentence_inds = sentence_inds[:self.max_sentence_len] if true_length >= self.max_sentence_len \
            else sentence_inds + [self.PAD_IND] * (self.max_sentence_len - true_length)
        # NOTICE:true_length = min(true_length, self.max_sentence_len)
        # because `bidirectional_dynamic_rnn` need param:seqence_length <= max_sentence_len
        return np.array([sentence_inds], dtype=np.int32), np.array([min(true_length, self.max_sentence_len)], dtype=np.int32)


if __name__ == '__main__':
    #定义输入辅助类
    max_sentence_len = 40
    pdh = PreDataHelper('../../model/unknown_model/unkVSres_char2ind_dct.pkl',
                        max_sentence_len)
    #构建ClassActivateMap类初始化参数
    input_tensor_name_lst = ['input_text:0', 'input_sentence_len:0', 'dropout_keep_prob:0']
    conv_name2filtersize_dct = dict([('conv-maxpool-{}/relu:0'.format(filtersize), filtersize) for filtersize in [1, 2, 3]])
    logits_name = 'output/logit:0'
    #ClassActivateMap初始化
    cam = ClassActivateMap(ckpt_fp='../../model/unknown_model/checkpoints_unkVSres_transformer_01',
                           label2id_dct={'unknown':0, 'known':1},
                           input_tensor_name_lst=input_tensor_name_lst,
                           conv_layer_tensor_name2filter_size_dct=conv_name2filtersize_dct,
                           logits_tensor_name=logits_name,)
    #输入文本
    text = '北京哪里好玩呢'
    #构建输入值，为字级别输入序列
    sentence_ids, true_length = pdh.char2id(text)
    input_value_lst = [sentence_ids, true_length, 1.0]
    #获取cam结果
    res_cam = cam.get_text_final_cam(input_value_lst, 'top', max_sentence_len, true_length[0])
    print(res_cam)
    #可视化，因示例任务的输入包含起始符<bos>和终止符<eos>，所以分别在前后加上加上
    cam_visualization(res_cam, text_fragment_lst=['<bos>']+list(text)+['<eos>'])
    ```
