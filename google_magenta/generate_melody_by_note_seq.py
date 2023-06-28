from note_seq.protobuf import music_pb2
import note_seq
from magenta.models.melody_rnn import melody_rnn_sequence_generator
from magenta.models.shared import sequence_generator_bundle
from note_seq.protobuf import generator_pb2
from note_seq.protobuf import music_pb2
import openai

def parse_midi_notes(response_text):
    lines = response_text.strip().split("\n")
    notes = []
    
    for line in lines:
        note_data = line.strip().split(",")
        
        if len(note_data) == 5:
            note = {
                "pitch": int(note_data[1].strip()),
                "start_time": float(note_data[2].strip()),
                "end_time": float(note_data[3].strip()),
                "velocity": int(note_data[4].strip())
            }
            
            notes.append(note)
    
    return notes

api_key = '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'
openai.api_key = api_key
model_id = 'gpt-3.5-turbo'


description = input("Enter the music piece description: ")
gptprompt = f"Generate MIDI notes for a {description}.\n\nPlease provide the notes in the following structure:\n\nnew_note_starting_here_indicator,pitch,start_time,end_time,velocity\n\nFor example:\n\nn,62,0.0,0.5,80\nn,60,0.5,1.0,80\n"

conversation = []
conversation.append({'role': 'system', 'content': 'I create great music just from a text description! Do you have any requests?'})
conversation.append({'role': 'user', 'content': gptprompt})

response = openai.ChatCompletion.create(
    model=model_id,
    messages=conversation
)

print(response.choices[0].message.content)

parsed_notes = parse_midi_notes(response.choices[0].message.content)

chatGPTSong = music_pb2.NoteSequence()

for note in parsed_notes:
    chatGPTSong.notes.add(
        pitch=note["pitch"],
        start_time=note["start_time"],
        end_time=note["end_time"],
        velocity=note["velocity"]
    )

for note in parsed_notes:
    pitch = note["pitch"]
    start_time = note["start_time"]
    end_time = note["end_time"]
    velocity = note["velocity"]
    
    note_info = f"Pitch: {pitch}, Start Time: {start_time}, End Time: {end_time}, Velocity: {velocity}"
    print(note_info)


# Find the highest end_time value
highest_end_time = max(note["end_time"] for note in parsed_notes)
chatGPTSong.total_time = highest_end_time
chatGPTSong.tempos.add(qpm=60)

# Initialize the model.
print("Initializing Melody RNN...")
bundle = sequence_generator_bundle.read_bundle_file('melody_rnn_bundles/attention_rnn.mag')
generator_map = melody_rnn_sequence_generator.get_generator_map()
melody_rnn = generator_map['attention_rnn'](checkpoint=None, bundle=bundle)
melody_rnn.initialize()

print('🎉 Done!')

input_sequence = chatGPTSong # change this to teapot if you want
num_steps = 128 # change this for shorter or longer sequences
temperature = 0.5 # the higher the temperature the more random the sequence.

# Set the start time to begin on the next step after the last note ends.
last_end_time = (max(n.end_time for n in input_sequence.notes)
                  if input_sequence.notes else 0)
qpm = input_sequence.tempos[0].qpm 
seconds_per_step = 60.0 / qpm / melody_rnn.steps_per_quarter
total_seconds = num_steps * seconds_per_step

generator_options = generator_pb2.GeneratorOptions()
generator_options.args['temperature'].float_value = temperature
generate_section = generator_options.generate_sections.add(
  start_time=last_end_time + seconds_per_step,
  end_time=total_seconds)

# Ask the model to continue the sequence.
sequence = melody_rnn.generate(input_sequence, generator_options)

note_seq.play_sequence(sequence, synth=note_seq.synthesize)
note_seq.sequence_proto_to_midi_file(sequence, 'output/guitar_sample_output.midi')