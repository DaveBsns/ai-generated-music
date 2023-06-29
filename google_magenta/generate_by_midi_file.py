from note_seq.protobuf import music_pb2
import note_seq
from magenta.models.melody_rnn import melody_rnn_sequence_generator
from magenta.models.shared import sequence_generator_bundle
from note_seq.protobuf import generator_pb2
from note_seq.protobuf import music_pb2
# Load the MIDI file
# midi_file = note_seq.midi_file_to_sequence_proto('./google_magenta/midi_input/sweet_home_alabama.mid')
midi_file = note_seq.midi_file_to_sequence_proto('./google_magenta/midi_input/hes_a_pirate.mid')
# Add the notes from the MIDI file to the NoteSequence
sweet_home_alabama = music_pb2.NoteSequence()
for note in midi_file.notes:
  sweet_home_alabama.notes.add(pitch=note.pitch, start_time=note.start_time, end_time=note.end_time, velocity=note.velocity)

# Set the total time of the sequence
sweet_home_alabama.total_time = max(note.end_time for note in sweet_home_alabama.notes)
print(sweet_home_alabama.total_time)

# Add a tempo to the sequence
sweet_home_alabama.tempos.add(qpm=midi_file.tempos[0].qpm)


print("Initializing Melody RNN...")
bundle = sequence_generator_bundle.read_bundle_file('./google_magenta/melody_rnn_bundles/basic_rnn.mag')
generator_map = melody_rnn_sequence_generator.get_generator_map()
melody_rnn = generator_map['basic_rnn'](checkpoint=None, bundle=bundle)
melody_rnn.initialize()

print('🎉 Done!')


input_sequence = sweet_home_alabama
num_steps = 48000 # change this for shorter or longer sequences
temperature = 100.0 # the higher the temperature the more random the sequence.

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


print("Number of notes in input sequence:", len(input_sequence.notes))
print("Value of num_steps:", num_steps)




# Ask the model to continue the sequence.
sequence = melody_rnn.generate(input_sequence, generator_options)

# note_seq.play_sequence(sequence, synth=note_seq.synthesize)
# note_seq.sequence_proto_to_midi_file(sequence, './output/like_a_sweet_home.mid')

print("Number of notes in input sequence:", len(sweet_home_alabama.notes))


note_seq.play_sequence(sweet_home_alabama, synth=note_seq.synthesize)
note_seq.sequence_proto_to_midi_file(sweet_home_alabama, './google_magenta/output/like_a_sweet_home.mid')
'''
'''