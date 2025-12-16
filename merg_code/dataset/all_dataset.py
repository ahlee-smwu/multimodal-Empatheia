import copy
import os
import re
import torch
import numpy as np
import json
import phonemizer
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pandas as pd
import glob


def transform_conv_id(id):
    return re.sub(r'^0+', '', id)


def calculate_pad_mask(style_clip, dim):
    is_padding = (style_clip == 0).all(dim=dim)
    mask = is_padding
    return mask


class TextCleaner:
    # IPA Phonemizer: https://github.com/bootphon/phonemizer
    def __init__(self, dummy=None):
        _pad = "$"
        _punctuation = ';:,.!?¡¿—…"«»“” '
        _letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
        _letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
        # Export all symbols:
        symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)
        dicts = {}
        for i in range(len((symbols))):
            dicts[symbols[i]] = i
        self.word_index_dictionary = dicts
        print(len(dicts))

    def __call__(self, text):
        indexes = []
        for char in text:
            try:
                indexes.append(self.word_index_dictionary[char])
            except KeyError:
                print(text)
        return indexes


class multimodal_empathetic_dialogue(Dataset):
    def __init__(self, args):
        super(multimodal_empathetic_dialogue, self).__init__()
        self.args = args
        # ★ ADD THIS BLOCK RIGHT HERE
        models_cfg = args.get('models', {})
        self.audio_path = self.args.get('audio_path', None)
        self.video_path = self.args.get('video_path', None)
        # ★ END NEW BLOCK
        self.age_projection = {
            "child": 0,
            "young": 1,
            "middle-aged": 2,
            "elderly": 3
        }
        self.gender_projection = {
            "male": 0,
            "female": 1
        }
        self.timbre_projection = {
            "high": 0,
            "mid": 1,
            "low": 2
        }

        combinations = []
        self.profile_projection = {}
        label = 0
        for age in self.age_projection.keys():
            for gender in self.gender_projection.keys():
                for timbre in self.timbre_projection.keys():
                    combination = f"{age}_{gender}_{timbre}"
                    combinations.append(combination)
                    self.profile_projection[combination] = label
                    label += 1

        self.ed_emotion_projection = {
            'conflicted': 'anxious',
            'vulnerability': 'afraid',
            'helplessness': 'afraid',
            'sadness': 'sad',
            'pensive': 'sentimental',
            'frustration': 'annoyed',
            'weary': 'tired',
            'anxiety': 'anxious',
            'reflective': 'sentimental',
            'upset': 'disappointed',
            'worried': 'anxious',
            'fear': 'afraid',
            'frustrated': 'sad',
            'fatigue': 'tired',
            'lost': 'jealous',
            'disappointment': 'disappointed',
            'nostalgia': 'nostalgic',
            'exhaustion': 'tired',
            'uneasy': 'anxious',
            'loneliness': 'lonely',
            'fragile': 'afraid',
            'confused': 'jealous',
            'vulnerable': 'afraid',
            'thoughtful': 'sentimental',
            'stressed': 'anxious',
            'concerned': 'anxious',
            'tiredness': 'tired',
            'burdened': 'anxious',
            'melancholy': 'sad',
            'overwhelmed': 'anxious',
            'worry': 'anxious',
            'heavy-hearted': 'sad',
            'melancholic': 'sad',
            'nervous': 'anxious',
            'fearful': 'afraid',
            'stress': 'anxious',
            'confusion': 'anxious',
            'inadequacy': 'ashamed',
            'regret': 'guilty',
            'helpless': 'afraid',
            'concern': 'anxious',
            'exhausted': 'tired',
            'overwhelm': 'anxious',
            'tired': 'tired',
            'disappointed': 'sad',
            'surprised': 'surprised',
            'excited': 'happy',
            'angry': 'angry',
            'proud': 'happy',
            'annoyed': 'angry',
            'grateful': 'happy',
            'lonely': 'sad',
            'afraid': 'fear',
            'terrified': 'fear',
            'guilty': 'sad',
            'impressed': 'surprised',
            'disgusted': 'disgusted',
            'hopeful': 'happy',
            'confident': 'happy',
            'furious': 'angry',
            'anxious': 'sad',
            'anticipating': 'happy',
            'joyful': 'happy',
            'nostalgic': 'sad',
            'prepared': 'happy',
            'jealous': 'contempt',
            'content': 'happy',
            'devastated': 'surprised',
            'embarrassed': 'sad',
            'caring': 'happy',
            'sentimental': 'sad',
            'trusting': 'happy',
            'ashamed': 'sad',
            'apprehensive': 'fear',
            'faithful': 'happy'
        }

        self.emotion_projection = {
            "happy": 0,
            "surprised": 1,
            "angry": 2,
            "fear": 3,
            "sad": 4,
            "disgusted": 5,
            "contempt": 6
        }

        self.data = []
        with open(os.path.join(args['models']['data_path'], args['mode'] + '.json'), 'r', encoding='utf-8') as f:
            self.raw_data = json.load(f)

        # self.audio_path = args['models'].get('audio_path', None)
        # self.video_path = args['models'].get('video_path', None)

        if args['mode'] == 'train':
            for item in tqdm(self.raw_data, total=len(self.raw_data)):
                turn = item['turns'][-1]
                conversation_id = item['conversation_id']
                speaker_profile = item['speaker_profile']
                listener_profile = item['listener_profile']
                topic = item['topic']
                self.data.append({
                    'conversation_id': conversation_id,
                    'turn': turn,
                    'speaker_profile': speaker_profile,
                    'listener_profile': listener_profile,
                    'topic': topic,
                })
        else:  # test
            for item in tqdm(self.raw_data, total=len(self.raw_data)):
                if item['turns']:
                    turn = item['turns'][-1]
                    conversation_id = item['conversation_id']
                    speaker_profile = item['speaker_profile']
                    listener_profile = item['listener_profile']
                    topic = item['topic']

                    self.data.append({
                        'conversation_id': conversation_id,
                        'turn': turn,
                        'speaker_profile': speaker_profile,
                        'listener_profile': listener_profile,
                        'topic': topic,
                    })
                else:  # test
                    for item in tqdm(self.raw_data, total=len(self.raw_data)):
                        if item['turns']:
                            turn = item['turns'][-1]
                            conversation_id = item['conversation_id']
                            self.data.append({
                                'conversation_id': conversation_id,
                                'turn': turn,
                            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        dia_id = transform_conv_id(item['conversation_id'])
        length = len(item['turn']['dialogue_history'])
        self.audio_path = self.args.get('audio_path', None)
        self.video_path = self.args.get('video_path', None)

        response_utt_name = f'dia{dia_id}utt{length}'
        response_audio = None
        response_video = None
        if self.audio_path is not None:
            response_audio = glob.glob(os.path.join(self.audio_path, f"{response_utt_name}_*.wav"))
        if self.video_path is not None:
            response_video = glob.glob(os.path.join(self.video_path, f"{response_utt_name}_*.mp4"))

        emotion = item['turn']['chain_of_empathy'].get('speaker_emotion', [])
        if isinstance(emotion, list):
            emotion = emotion[0] if len(emotion) > 0 else ''
        emotion = self.ed_emotion_projection.get(emotion, emotion)
        emotion = self.emotion_projection.get(emotion, -1)

        data = {
            'dia_id': dia_id,
            'context': item['turn'].get('context', ''),
            'dialogue_history': item['turn'].get('dialogue_history', []),
            'response': item['turn'].get('response', ''),
            'chain_of_empathy': item['turn'].get('chain_of_empathy', {}),
            # 'response_emotion': self.emotion_projection[item['turn']['chain_of_empathy'].get('speaker_emotion', '')],
            'response_emotion': [emotion],
            'response_age': self.age_projection[item['listener_profile']['age']],
            'response_gender': self.gender_projection[item['listener_profile']['gender']],
            'response_timbre': self.timbre_projection[item['listener_profile']['timbre']],
            'profile_id': self.profile_projection[
                item['listener_profile']['age'] + '_' + item['listener_profile']['gender'] + '_' +
                item['listener_profile']['timbre']],
            # ★ NEW: raw tensors that Stage3 SAL will use
            'response_audio': response_audio,  # torch.Tensor or None
            'response_video': response_video,  # torch.Tensor or None
        }
        return data

    def load_tensor(self, path, response_utt_name):
        file_path = os.path.join(path, f"{response_utt_name}.pt")
        # print(f"Loading from: {file_path}")
        try:
            tensor_data = torch.load(file_path, map_location='cpu').get(response_utt_name)
            if tensor_data is None:
                raise ValueError(f"{response_utt_name} not found in {file_path}")
            return tensor_data
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None

    def collate_fn(self, batch):
        conversations = []
        dia_ids = [instance['dia_id'] for instance in batch]
        response_age = [instance['response_age'] for instance in batch]
        response_gender = [instance['response_gender'] for instance in batch]
        response_timbre = [instance['response_timbre'] for instance in batch]
        response_emotion = [instance['response_emotion'] for instance in batch]
        response_profile = [instance['profile_id'] for instance in batch]
        # ★ NEW: collect wav & video
        response_audio = [instance['response_audio'] for instance in batch]
        response_video = [instance['response_video'] for instance in batch]

        dialogue_history = [instance['dialogue_history'] for instance in batch]
        response = [instance['response'] for instance in batch]
        coe = [instance['chain_of_empathy'] for instance in batch]
        # profile_ids = [instance['profile_id'] for instance in batch]
        assert len(dia_ids) == len(dialogue_history) == len(response) == len(coe)
        for index in range(len(dia_ids)):
            conversations.append(
                {
                    'dialogue_history': dialogue_history[index],
                    'response': response[index],
                    'coe': coe[index]
                }
            )

        return {'dia_ids': dia_ids,
                'conversations': conversations,
                'response_age': response_age,
                'response_emotion': response_emotion,
                'response_gender': response_gender,
                'response_timbre': response_timbre,
                'response_profile': response_profile,
                # ★ NEW: what Stage3 SAL is screaming for
                'response_audio': response_audio,
                'response_video': response_video,
                }


''' print(batch)
{'dia_ids': ['17677'],
 'conversations': [{'dialogue_history': [{'index': 0,
     'role': 'speaker',
     'utterance': "I just feel like I'm constantly juggling everything and it's wearing me down."},
    {'index': 1,
     'role': 'listener',
     'utterance': 'That sounds really tough. It must be exhausting to manage so much all at once.'},
    {'index': 2,
     'role': 'speaker',
     'utterance': "It's like no matter how hard I try, I can't seem to find a balance."},
    {'index': 3,
     'role': 'listener',
     'utterance': 'Finding that balance can be really challenging, especially with so many expectations.'},
    {'index': 4,
     'role': 'speaker',
     'utterance': "I just wish I could catch a break and feel like I'm on top of things again."}],
   'response': "Everyone has moments like these, and it's okay to ask for help when you need it.",
   'coe': {'speaker_emotion': 'anxious',
    'event_scenario': 'Feeling the need for a break',
    'emotion_cause': 'The pressure of managing responsibilities without relief',
    'goal_to_response': 'To find reassurance that seeking help is acceptable'}}],
 'response_age': [2],
 'response_emotion': [4],
 'response_gender': [0],
 'response_timbre': [2],
 'response_profile': [14],
 'response_wav': [None],
 'response_video': [None]}
'''