import torch

class SeqToSeq(torch.nn.Module):

	def __init__(self, argdict):
		super().__init__()
		self.argdict=argdict
		self.embeddings=torch.nn.Embedding(self.argdict['input_size'], self.argdict['embed_size'])
		self.rnn_encoder=torch.nn.GRU(self.argdict['embed_size'], self.argdict['hidden_size'], 2, batch_first=True, bidirectional=False)
		"""Decoder"""
		self.embeddings=torch.nn.Embedding(self.argdict['input_size'], self.argdict['embed_size'])
		self.rnn_decoder=torch.nn.GRU(self.argdict['embed_size'], self.argdict['hidden_size'], 1, batch_first=True, bidirectional=False)

		self.loss=torch.nn.CrossEntropyLoss(ignore_index=argdict['pad_idx'])

	def forward(self, input_seq, output_seq):
		embed_in=self.embeddings(input_seq)
		_, hidden=self.rnn_encoder(embed_in)
		embed_out=self.embeddings(output_seq[:-1])
		outputs, _=self.rnn_decoder(embed_out)

		target=output_seq[1:]
		if output_seq is not None:
			print(outputs.shape)
			print(target.shape)
			loss=self.loss(outputs, target)

		return {'logits':outputs, 'loss':loss}

