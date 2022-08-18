import torch

class SeqToSeq(torch.nn.Module):
	"""Simple seq to seq model"""

	def __init__(self, argdict):
		super().__init__()
		self.argdict=argdict
		self.embeddings=torch.nn.Embedding(self.argdict['input_size'], self.argdict['embed_size'])
		self.rnn_encoder=torch.nn.GRU(self.argdict['embed_size'], self.argdict['hidden_size'], 1, batch_first=True, bidirectional=False)
		"""Decoder"""
		self.embeddings=torch.nn.Embedding(self.argdict['input_size'], self.argdict['embed_size'])
		self.rnn_decoder=torch.nn.GRU(self.argdict['embed_size'], self.argdict['hidden_size'], 1, batch_first=True, bidirectional=False)
		self.output_to_vocab=torch.nn.Linear(self.argdict['hidden_size'], self.argdict['input_size'])

		self.loss=torch.nn.CrossEntropyLoss(ignore_index=argdict['pad_idx'])

	def forward(self, input_seq, output_seq):
		embed_in=self.embeddings(input_seq)
		_, hidden=self.rnn_encoder(embed_in)
		#Take the last hidden step as input for the decoder
		embed_out=self.embeddings(output_seq[:, :-1])
		outputs, _=self.rnn_decoder(embed_out, hidden)
		outputs=self.output_to_vocab(outputs)

		target=output_seq[:, 1:]
		print(outputs.view(-1, outputs.shape[-1]).shape)
		print(target.view(-1.shape))
		if output_seq is not None:
			loss=self.loss(outputs.view(-1, outputs.shape[-1]), target.view(-1))
			return {'logits':outputs, 'loss':loss}
		else:
			return {'logits':outputs}

	def generate(self, input_seq, num_beams=10, num_return_sequences=1, max_length=50, device='cpu'):
		curr=torch.zeros((10, 1))+self.argdict['bos_idx']
		# curr_log_prob=torch.zeros((10, 1))
		curr_log_prob=torch.Tensor([0,1,2,3,4,5,6,7,8,9])
		curr_log_prob=curr_log_prob.unsqueeze(1).to(device)
		curr=curr.int().to(device)
		embed_in=self.embeddings(input_seq)
		_, hidden=self.rnn_encoder(embed_in)
		for i in range(max_length):
			embed_out=self.embeddings(curr)
			outputs, _ = self.rnn_decoder(embed_out, hidden)
			outputs = self.output_to_vocab(outputs).squeeze(1)
			outputs=torch.nn.functional.log_softmax(outputs, dim=-1)
			vocab_output=outputs.shape[-1]
			curr_log_prob=curr_log_prob.repeat(1, vocab_output)
			#This denotes the probability for the last token. Add this probability to the log probability of the preceding sentence
			phrase_log_prob=curr_log_prob+outputs
			phrase_log_prob=phrase_log_prob.view(1, -1)
			top=torch.topk(phrase_log_prob, k=num_beams, dim=-1)
			#Update the curr log prob
			#next curr is going to be shaped

			#First we need to find for each topk from which branch it came
			for value, index in zip(top.values.squeeze(0), top.indices.squeeze(0)):
				#We need to find from which branch it comes
				x=index.item()//vocab_output
				y=index.item()%vocab_output
				print(index.item())
				print(x, y)

			print(top)
			fds
		print(curr)
		output=self.forward(curr, None)
		pass
