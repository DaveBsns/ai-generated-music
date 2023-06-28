from note_seq.protobuf import music_pb2
import note_seq
from magenta.models.melody_rnn import melody_rnn_sequence_generator
from magenta.models.shared import sequence_generator_bundle
from note_seq.protobuf import generator_pb2
from note_seq.protobuf import music_pb2

'''
guitar = music_pb2.NoteSequence()
guitar.notes.add(pitch=40, start_time=0, end_time=0.25, is_drum=False, instrument=26, velocity=80)
guitar.notes.add(pitch=42, start_time=0.25, end_time=0.5, is_drum=False, instrument=26, velocity=80)
guitar.notes.add(pitch=45, start_time=0.5, end_time=0.75, is_drum=False, instrument=26, velocity=80)
guitar.notes.add(pitch=40, start_time=0.75, end_time=1, is_drum=False, instrument=26, velocity=80)
guitar.notes.add(pitch=42, start_time=1, end_time=1.25, is_drum=False, instrument=26, velocity=80)
guitar.notes.add(pitch=45, start_time=1.25, end_time=1.5, is_drum=False, instrument=26, velocity=80)
guitar.notes.add(pitch=40, start_time=1.5, end_time=1.75, is_drum=False, instrument=26, velocity=80)
guitar.notes.add(pitch=42, start_time=1.75, end_time=2, is_drum=False, instrument=26, velocity=80)
guitar.notes.add(pitch=45, start_time=2, end_time=2.25, is_drum=False, instrument=26, velocity=80)
guitar.notes.add(pitch=40, start_time=2.25, end_time=2.5, is_drum=False, instrument=26, velocity=80)
guitar.notes.add(pitch=42, start_time=2.5, end_time=2.75, is_drum=False, instrument=26, velocity=80)
guitar.notes.add(pitch=45, start_time=2.75, end_time=3, is_drum=False, instrument=26, velocity=80)
guitar.notes.add(pitch=40, start_time=3, end_time=3.25, is_drum=False, instrument=26, velocity=80)
guitar.notes.add(pitch=42, start_time=3.25, end_time=3.5, is_drum=False, instrument=26, velocity=80)
guitar.notes.add(pitch=45, start_time=3.5, end_time=3.75, is_drum=False, instrument=26, velocity=80)
guitar.notes.add(pitch=40, start_time=3.75, end_time=4, is_drum=False, instrument=26, velocity=80)
guitar.total_time = 4

guitar.tempos.add(qpm=60)


note_seq.play_sequence(guitar,synth=note_seq.synthesize)


# This creates a file called `drums_sample_output.mid`, containing the drums solo we've been using.
note_seq.sequence_proto_to_midi_file(guitar, '../output/guitar_sample_output.mid')
'''



sweet_home_alabama = music_pb2.NoteSequence()

# Add the notes to the sequence
sweet_home_alabama.notes.add(pitch=62, start_time=0.0, is_drum=True, instrument=10, end_time=0.5, velocity=80)
sweet_home_alabama.notes.add(pitch=60, start_time=0.5, is_drum=True, instrument=10,end_time=1.0, velocity=80)
sweet_home_alabama.notes.add(pitch=57, start_time=1.0, is_drum=True, instrument=10,end_time=1.5, velocity=80)
sweet_home_alabama.notes.add(pitch=55, start_time=1.5, is_drum=True, instrument=10,end_time=2.0, velocity=80)
sweet_home_alabama.notes.add(pitch=53, start_time=2.0, is_drum=True, instrument=10,end_time=2.5, velocity=80)
sweet_home_alabama.notes.add(pitch=50, start_time=2.5, is_drum=True, instrument=10,end_time=3.0, velocity=80)
sweet_home_alabama.notes.add(pitch=48, start_time=3.0, is_drum=True, instrument=10,end_time=4.0, velocity=80)
sweet_home_alabama.notes.add(pitch=50, start_time=4.0, is_drum=True, instrument=10,end_time=4.5, velocity=80)
sweet_home_alabama.notes.add(pitch=53, start_time=4.5, is_drum=True, instrument=10,end_time=5.0, velocity=80)
sweet_home_alabama.notes.add(pitch=52, start_time=5.0, is_drum=True, instrument=10,end_time=5.5, velocity=80)
sweet_home_alabama.notes.add(pitch=48, start_time=5.5, is_drum=True, instrument=10,end_time=6.0, velocity=80)
sweet_home_alabama.notes.add(pitch=48, start_time=6.0, is_drum=True, instrument=10,end_time=6.5, velocity=80)
sweet_home_alabama.notes.add(pitch=50, start_time=6.5, is_drum=True, instrument=10,end_time=7.0, velocity=80)
sweet_home_alabama.notes.add(pitch=55, start_time=7.0, is_drum=True, instrument=10,end_time=8.0, velocity=80)

# Set the total time of the sequence
sweet_home_alabama.total_time = 8

# Add a tempo to the sequence
sweet_home_alabama.tempos.add(qpm=60)
# Initialize the model.
print("Initializing Melody RNN...")
bundle = sequence_generator_bundle.read_bundle_file('../melody_rnn_bundles/attention_rnn.mag')
generator_map = melody_rnn_sequence_generator.get_generator_map()
melody_rnn = generator_map['attention_rnn'](checkpoint=None, bundle=bundle)
melody_rnn.initialize()

print('🎉 Done!')

input_sequence = sweet_home_alabama # change this to teapot if you want
num_steps = 128 # change this for shorter or longer sequences
temperature = 1.0 # the higher the temperature the more random the sequence.

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
note_seq.sequence_proto_to_midi_file(sequence, '../output/guitar_sample_output.mid')