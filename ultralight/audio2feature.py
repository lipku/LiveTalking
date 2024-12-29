from transformers import Wav2Vec2Processor, HubertModel
import torch
import numpy as np


class Audio2Feature():
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
        self.model = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft").to(self.device)


    @torch.no_grad()
    def get_hubert_from_16k_speech(self, speech):
        if speech.ndim == 2:
            speech = speech[:, 0]  # [T, 2] ==> [T,]
        input_values_all = self.processor(speech, return_tensors="pt", sampling_rate=16000).input_values  # [1, T]
        input_values_all = input_values_all.to(self.device)
	    
        kernel = 400
        stride = 320
        clip_length = stride * 1000
        num_iter = input_values_all.shape[1] // clip_length
        expected_T = (input_values_all.shape[1] - (kernel-stride)) // stride
        res_lst = []
        for i in range(num_iter):
            if i == 0:
                start_idx = 0
                end_idx = clip_length - stride + kernel
            else:
                start_idx = clip_length * i
                end_idx = start_idx + (clip_length - stride + kernel)
            input_values = input_values_all[:, start_idx: end_idx]
            hidden_states = self.model.forward(input_values).last_hidden_state  # [B=1, T=pts//320, hid=1024]
            res_lst.append(hidden_states[0])
        if num_iter > 0:
            input_values = input_values_all[:, clip_length * num_iter:]
        else:
            input_values = input_values_all
        if input_values.shape[1] >= kernel:  # if the last batch is shorter than kernel_size, skip it            
            hidden_states = self.model(input_values).last_hidden_state  # [B=1, T=pts//320, hid=1024]
        res_lst.append(hidden_states[0])
        ret = torch.cat(res_lst, dim=0).cpu()  # [T, 1024]
        assert abs(ret.shape[0] - expected_T) <= 1
        if ret.shape[0] < expected_T:
            ret = torch.nn.functional.pad(ret, (0,0,0,expected_T-ret.shape[0]))
        else:
            ret = ret[:expected_T]
        return ret

    def get_sliced_feature(self,
                           feature_array,
                           vid_idx,
                           audio_feat_length=[8,8],
                           fps=25):
        """
        Get sliced features based on a given index
        :param feature_array: 
        :param start_idx: the start index of the feature
        :param audio_feat_length:
        :return: 
        """
        length = len(feature_array)
        selected_feature = []
        selected_idx = []

        center_idx = int(vid_idx*50/fps)
        left_idx = center_idx-audio_feat_length[0]*2
        right_idx = center_idx + (audio_feat_length[1])*2

        for idx in range(left_idx,right_idx):
            idx = max(0, idx)
            idx = min(length-1, idx)
            x = feature_array[idx]
            selected_feature.append(x)
            selected_idx.append(idx)

        selected_feature = np.concatenate(selected_feature, axis=0)
        selected_feature = selected_feature.reshape(-1, 1024)
        return selected_feature,selected_idx

    def feature2chunks(self,feature_array,fps,batch_size,audio_feat_length = [8,8],start=0):
        whisper_chunks = []
        whisper_idx_multiplier = 50./fps
        i = 0
        #print(f"video in {fps} FPS, audio idx in 50FPS")
        for _ in range(batch_size):
            # start_idx = int(i * whisper_idx_multiplier)
            # if start_idx>=len(feature_array):
            #     break
            selected_feature,selected_idx = self.get_sliced_feature(feature_array= feature_array,vid_idx = i+start,audio_feat_length=audio_feat_length,fps=fps)
            #print(f"i:{i},selected_idx {selected_idx}")
            whisper_chunks.append(selected_feature)
            i += 1

        return whisper_chunks
