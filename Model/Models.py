import torch

class SeqToSeq(torch.nn.Module):
	"""Simple seq to seq model"""

	def __init__(self, argdict):
		super().__init__()
		self.argdict=argdict
		self.embeddings=torch.nn.Embedding(self.argdict['input_size'], self.argdict['embed_size'])
		self.rnn_encoder=torch.nn.GRU(self.argdict['embed_size'], self.argdict['hidden_size'], 2, batch_first=True, bidirectional=False)
		"""Decoder"""
		self.embeddings=torch.nn.Embedding(self.argdict['input_size'], self.argdict['embed_size'])
		self.rnn_decoder=torch.nn.GRU(self.argdict['embed_size'], self.argdict['hidden_size'], 1, batch_first=True, bidirectional=False)
		self.output_to_vocab=torch.nn.Linear(self.argdict['hidden_size'], self.argdict['input_size'])

		self.loss=torch.nn.CrossEntropyLoss(ignore_index=argdict['pad_idx'])

	def forward(self, input_seq, output_seq):
		embed_in=self.embeddings(input_seq)
		_, hidden=self.rnn_encoder(embed_in)
		embed_out=self.embeddings(output_seq[:-1])
		outputs, _=self.rnn_decoder(embed_out)
		outputs=self.output_to_vocab(outputs)

		target=output_seq[1:]
		if output_seq is not None:
			loss=self.loss(outputs.view(-1, outputs.shape[-1]), target.view(-1))
			return {'logits':outputs, 'loss':loss}
		else:
			return {'logits':outputs}

	def generate(self, input_seq, num_beams=10, num_return_sequences=1, max_length=50, device='cpu'):
		curr=torch.zeros((10, 1))+self.argdict['bos_idx']
		# curr_log_prob=torch.zeros((10, 1))
		curr_log_prob=torch.Tensor([0,1,2,3,4,5,6,7,8,9])
		curr=curr.int().to(device)
		embed_in=self.embeddings(input_seq)
		_, hidden=self.rnn_encoder(embed_in)
		for i in range(max_length):
			embed_out=self.embeddings(curr)
			outputs, _ = self.rnn_decoder(embed_out)
			outputs = self.output_to_vocab(outputs).squeeze(1)
			outputs=torch.nn.functional.log_softmax(outputs, dim=-1)
			print(outputs.shape)
			#This denotes the probability for the last token. Add this probability to the log probability of the preceding sentence
			phrase_log_prob=curr_log_prob+outputs
			print(outputs)
			print(phrase_log_prob)
			fds
			top=torch.topk(outputs, k=num_beams, dim=-1)
			print(top)
			fds
		print(curr)
		output=self.forward(curr, None)
		pass
