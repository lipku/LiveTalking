import os
from .whisper import load_model
import soundfile as sf
import numpy as np
import time
import sys
sys.path.append("..")

class Audio2Feature():
    def __init__(self, 
                 whisper_model_type="tiny",
                 model_path="./models/whisper/tiny.pt"):
        self.whisper_model_type = whisper_model_type
        self.model = load_model(model_path) #

    def get_sliced_feature(self,
                           feature_array, 
                           vid_idx, 
                           audio_feat_length=[2,2],
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
        right_idx = center_idx + (audio_feat_length[1]+1)*2
        
        for idx in range(left_idx,right_idx):
            idx = max(0, idx)
            idx = min(length-1, idx)
            x = feature_array[idx]
            selected_feature.append(x)
            selected_idx.append(idx)
        
        selected_feature = np.concatenate(selected_feature, axis=0)
        selected_feature = selected_feature.reshape(-1, 384)# 50*384
        return selected_feature,selected_idx

    def get_sliced_feature_sparse(self,feature_array, vid_idx, audio_feat_length= [2,2],fps = 25):
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

        for dt in range(-audio_feat_length[0],audio_feat_length[1]+1):
            left_idx = int((vid_idx+dt)*50/fps)
            if left_idx<1 or left_idx>length-1:
                left_idx = max(0, left_idx)
                left_idx = min(length-1, left_idx)

                x = feature_array[left_idx]
                x = x[np.newaxis,:,:]
                x = np.repeat(x, 2, axis=0)
                selected_feature.append(x)
                selected_idx.append(left_idx)
                selected_idx.append(left_idx)
            else:
                x = feature_array[left_idx-1:left_idx+1]
                selected_feature.append(x)
                selected_idx.append(left_idx-1)
                selected_idx.append(left_idx)
        selected_feature = np.concatenate(selected_feature, axis=0)
        selected_feature = selected_feature.reshape(-1, 384)# 50*384
        return selected_feature,selected_idx
    

    def feature2chunks(self,feature_array,fps,audio_feat_length = [2,2]):
        whisper_chunks = []
        whisper_idx_multiplier = 50./fps 
        i = 0
        print(f"video in {fps} FPS, audio idx in 50FPS")
        while 1:
            start_idx = int(i * whisper_idx_multiplier)
            selected_feature,selected_idx = self.get_sliced_feature(feature_array= feature_array,vid_idx = i,audio_feat_length=audio_feat_length,fps=fps)
            #print(f"i:{i},selected_idx {selected_idx}")
            whisper_chunks.append(selected_feature)
            i += 1
            if start_idx>len(feature_array):
                break

        return whisper_chunks

    def audio2feat(self,audio_path):
        # get the sample rate of the audio
        result = self.model.transcribe(audio_path)
        embed_list = []
        for emb in result['segments']:
            encoder_embeddings = emb['encoder_embeddings']
            encoder_embeddings = encoder_embeddings.transpose(0,2,1,3)
            encoder_embeddings = encoder_embeddings.squeeze(0)
            start_idx = int(emb['start'])
            end_idx = int(emb['end'])
            emb_end_idx = int((end_idx - start_idx)/2)
            embed_list.append(encoder_embeddings[:emb_end_idx])
        concatenated_array = np.concatenate(embed_list, axis=0)
        return concatenated_array

if __name__ == "__main__":
    audio_processor = Audio2Feature(model_path="../../models/whisper/whisper_tiny.pt")
    audio_path = "./test.mp3"
    array = audio_processor.audio2feat(audio_path)
    print(array.shape)
    fps = 25
    whisper_idx_multiplier = 50./fps 

    i = 0
    print(f"video in {fps} FPS, audio idx in 50FPS")
    while 1:
        start_idx = int(i * whisper_idx_multiplier)
        selected_feature,selected_idx = audio_processor.get_sliced_feature(feature_array= array,vid_idx = i,audio_feat_length=[2,2],fps=fps)
        print(f"video idx {i},\t audio idx {selected_idx},\t shape {selected_feature.shape}")
        i += 1
        if start_idx>len(array):
            break
