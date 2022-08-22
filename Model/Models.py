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
		if output_seq is not None:
			loss=self.loss(outputs.view(-1, outputs.shape[-1]), target.contiguous().view(-1))
			return {'logits':outputs, 'loss':loss}
		else:
			return {'logits':outputs}

	def generate(self, input_seq, num_beams=10, num_return_sequences=1, max_length=50, device='cpu'):
		bs=input_seq.shape[0]
		curr=torch.zeros((input_seq.shape[0], num_beams, 1))+self.argdict['bos_idx']
		# curr_log_prob=torch.zeros((10, 1))
		curr_log_prob=torch.zeros(bs, num_beams, 1)
		curr_log_prob=curr_log_prob.to(device)
		curr=curr.int().to(device)
		# print(input_seq.unsqueeze(1).expand(-1, 10, 1))#.view(input_seq.shape[0]*input_seq.shape[1], -1))
		input_seq=input_seq.unsqueeze(1).repeat(1, num_beams, 1).view(input_seq.shape[0]*num_beams, -1)
		embed_in=self.embeddings(input_seq)
		_, hidden=self.rnn_encoder(embed_in)
		for i in range(max_length):
			curr_reshaped=curr.view(curr.shape[0]*curr.shape[1], -1)
			embed_out=self.embeddings(curr_reshaped)
			outputs, _ = self.rnn_decoder(embed_out, hidden)
			outputs = self.output_to_vocab(outputs).squeeze(1)
			outputs=torch.nn.functional.log_softmax(outputs, dim=-1)
			vocab_output=outputs.shape[-1]
			curr_log_prob_reshaped=curr_log_prob.repeat(1, 1, vocab_output)
			#This denotes the probability for the last token. Add this probability to the log probability of the preceding sentence
			outputs=outputs.view(bs, num_beams, vocab_output)
			phrase_log_prob=curr_log_prob_reshaped+outputs
			phrase_log_prob=phrase_log_prob.view(bs, -1)
			top=torch.topk(phrase_log_prob, k=num_beams, dim=-1)
			values=top.values
			#Update the curr log prob
			#next curr is going to be shaped

			#First we need to find for each topk from which branch it came
			#The branch they come from in index//vocab_output, the vocab number will be index%vocab_output
			x=top.indices//vocab_output
			y=top.indices%vocab_output
			#We now need to combine both
			#
			new_log_prob=torch.zeros_like(curr_log_prob)
			new_index=torch.zeros((curr.shape[0], curr.shape[1], curr.shape[2]+1))
			for i, (og_branch, new_ind, log_prob_new) in enumerate(zip(x, y, values)):
				new_log_prob=curr_log_prob[i]+log_prob_new
			print(new_log_prob)
			fds
			#
			# for value, index in zip(top.values.squeeze(0), top.indices.squeeze(0)):
			# 	#We need to find from which branch it comes
			# 	x=index.item()//vocab_output
			# 	y=index.item()%vocab_output
			# 	print(index.item())
			# 	print(x, y)
			#
			# print(top)
			fds
		print(curr)
		output=self.forward(curr, None)
		pass
