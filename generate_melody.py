from magenta.models.melody_rnn import MelodyRnnModel
from magenta.models.melody_rnn import melody_rnn_sequence_generator
'''
model = MelodyRnnModel()
model.generate_melody()
'''

bundle = "./melody_rnn_bundles/basic_rnn.mag"
generator_map = melody_rnn_sequence_generator.get_generator_map()
melody_rnn = generator_map['basic_rnn'](checkpoint=None, bundle=bundle)
melody_rnn.initialize()
print("initialized!")