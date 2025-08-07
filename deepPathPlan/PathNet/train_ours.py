import os
import argparse
import torch
import torch.nn as nn
import os
from data_loader import pDataset 
from network import   trajFCNet
from einops import rearrange
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter  
from torch.utils.data import DataLoader, random_split
from loss import  NormalizeArcLoss,NonholoLoss,UniforArcLoss,CurvatureLoss,FullShapeCollisionLoss
from course import read_wei_course

def test_net(mpnet, val_loader, output_logfile, wei_arc, smooLoss, wei_pos, mseLoss, wei_rot,
             wei_hol, holoLoss, wei_uni, uniLoss,wei_cur,curLoss, wei_rsm,rotsLoss,wei_safety,safeLoss,
             wei_anchors,crossentropyloss,compt_count):
	mpnet.eval()
	avg_loss=0
	avg_posloss = 0
	avg_rotloss = 0
	avg_arcloss = 0
	avg_holloss = 0
	avg_uniloss = 0
	avg_curloss = 0
	avg_safeloss = 0
	avg_rsmloss = 0
	avg_anchorsloss = 0
	step_num = 0
	for batch in val_loader:
		with torch.no_grad():
			input = batch[0].cuda()
			labelops = batch[2].cuda()
			labelrot = batch[3].cuda()
			anchors = batch[4].cuda()
			
			labelgrids= rearrange(anchors, 'b c h w-> b c (h w)')
			labelgrids = rearrange(labelgrids, 'b c l-> (b c) l')
			labelgrids = torch.argmax(labelgrids, dim=1)

			opState, opRot,prbmap, predicted_anchors = mpnet(input, labelops,labelrot, anchors)




			arcloss = wei_arc * smooLoss(opState, labelops)
			posloss = wei_pos * torch.sqrt(mseLoss(opState, labelops))
			rotloss = wei_rot * torch.sqrt(mseLoss(opRot, labelrot))
			holloss = wei_hol * holoLoss(opState,opRot[:,1:,:])
			unilos = wei_uni * uniLoss(opState,labelops)
			curloss = wei_cur * curLoss(opState, opRot)

			rsmloss = wei_rsm * rotsLoss(opRot, labelrot)
			safetyloss = wei_safety * safeLoss(opState, opRot, input)
			
			gridloss = wei_anchors *  crossentropyloss(predicted_anchors,labelgrids)


			loss = posloss + arcloss + holloss + unilos + rotloss + curloss + rsmloss + safetyloss+gridloss
			avg_posloss = avg_posloss + posloss.item()
			avg_arcloss = avg_arcloss + arcloss.item()
			avg_holloss = avg_holloss + holloss.item()
			avg_uniloss = avg_uniloss + unilos.item()
			avg_rotloss = avg_rotloss + rotloss.item()
			avg_curloss = avg_curloss + curloss.item()
			avg_rsmloss = avg_rsmloss + rsmloss.item()
			avg_safeloss = avg_safeloss + safetyloss.item()
			avg_anchorsloss = avg_anchorsloss + gridloss.item()
			avg_loss=avg_loss+loss.item()
			step_num += 1


	output_logfile.write(f"--step: {compt_count}\n")
	output_logfile.write(f"--average loss: {avg_loss/step_num}\n")
	output_logfile.write(f"--average arcloss: {avg_arcloss/step_num}\n")
	output_logfile.write(f"--average holloss: {avg_holloss/step_num}\n")
	output_logfile.write(f"--average uniloss: {avg_uniloss/step_num}\n")
	output_logfile.write(f"--average posloss: {avg_posloss/step_num}\n")
	output_logfile.write(f"--average rotloss: {avg_rotloss/step_num}\n")
	output_logfile.write(f"--average curloss: {avg_curloss/step_num}\n")
	output_logfile.write(f"--average rsmloss: {avg_rsmloss/step_num}\n")
	output_logfile.write(f"--average safeloss: {avg_safeloss/step_num}\n")
	output_logfile.write(f"--average anchors: {avg_anchorsloss/step_num}\n")

def main(args):
	# Create model directory
	if not os.path.exists(args.model_path):
		os.makedirs(args.model_path)
  
	# 0. Build the models
	mpnet = trajFCNet(4,200,7,l=1.2,use_groundTruth=False)
	model_path='model.pkl'
	# mpnet.load_state_dict(torch.load("./models/"+model_path))
	if torch.cuda.is_available():
		mpnet.cuda()
	
	# 1. Loss and Optimizer
	mseLoss = nn.MSELoss()
	smooLoss = NormalizeArcLoss()
	rotsLoss = NormalizeArcLoss()
	holoLoss = NonholoLoss()
	uniLoss = UniforArcLoss()
	curLoss = CurvatureLoss()
	safeLoss = FullShapeCollisionLoss()
	crossentropyloss=nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(mpnet.parameters(), lr=args.learning_rate)	
	dataset = pDataset()
	writer = SummaryWriter('./path/to/log')
	# 2. Split into train / validation partitions
	n_val = int(len(dataset) * 0.001)
	n_train = len(dataset) - n_val
	train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(1))
	# 3. Create data loaders
	loader_args = dict(batch_size=args.batch_size, num_workers=8, pin_memory=True)
	train_loader = DataLoader(train_set, shuffle=True, **loader_args)
	val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)
	# Train the Models
	print("num_train:", n_train)
	


	step_count, compt_count = 0, 0
	wei_arc = torch.tensor(0.0)
	wei_uni = torch.tensor(0.0)
	wei_pos = torch.tensor(0.0)
	wei_hol = torch.tensor(0.0)
	wei_rot = torch.tensor(0.0)
	wei_cur = torch.tensor(0.0)
	wei_safety = torch.tensor(0.0)
	wei_rsm = torch.tensor(0.0)
	wei_anchors = torch.tensor(0.0)


	
	output_logfile = open("./models/"+model_path+'.txt', 'w')  
	output_logfile2 = open("./models/"+model_path+'Step.txt', 'w')  
	for epoch in range(0, args.num_epochs):
		if(epoch < 2):
			for param_group in optimizer.param_groups:
				param_group['lr'] = 1.0e-4
		else:
			for param_group in optimizer.param_groups:
				param_group['lr'] = 1.0e-5
		print("epoch" + str(epoch) + ' lr: ' + str(optimizer.state_dict()['param_groups'][0]['lr']))
		output_logfile.write("epoch" + str(epoch) + ' lr: ' + str(optimizer.state_dict()['param_groups'][0]['lr'])+'\n')
		wei_anchors, wei_arc, wei_uni, wei_pos, wei_hol, wei_rot, wei_cur, wei_safety, wei_rsm = read_wei_course(epoch)
		print('wei_arc: ', wei_arc)
		print('wei_uni: ', wei_uni)
		print('wei_pos: ', wei_pos)
		print('wei_hol: ', wei_hol)
		print('wei_rot: ', wei_rot)
		print('wei_cur: ', wei_cur)
		print('wei_safety: ',wei_safety)
		print('wei_rsm: ', wei_rsm)
		print('wei aors: ',	wei_anchors)	
		output_logfile.write('wei_arc: ' + str(wei_arc) + '\n')
		output_logfile.write('wei_uni: ' + str(wei_uni) + '\n')
		output_logfile.write('wei_pos: ' + str(wei_pos) + '\n')
		output_logfile.write('wei_hol: ' + str(wei_hol) + '\n')
		output_logfile.write('wei_rot: ' + str(wei_rot) + '\n')
		output_logfile.write('wei_cur: ' + str(wei_cur) + '\n')
		output_logfile.write('wei_safety: ' + str(wei_safety) + '\n')
		output_logfile.write('wei_rsm: ' + str(wei_rsm) + '\n')
		output_logfile.write('wei_aors: ' + str(wei_anchors) + '\n')
		avg_loss=0
		avg_posloss = 0
		avg_rotloss = 0
		avg_arcloss = 0
		avg_holloss = 0
		avg_uniloss = 0
		avg_curloss = 0
		avg_safeloss = 0
		avg_rsmloss = 0
		avg_anchorsloss = 0.0
		step_num = 0
		mpnet.train()


		with tqdm(total=n_train, desc=f'Epoch {epoch}/{args.num_epochs}', unit='data') as pbar:
			for batch in train_loader:
				mpnet.train()
				input = batch[0].cuda()
				raw_env = batch[1].cuda()
				labelops = batch[2].cuda()
				labelrot = batch[3].cuda()
				anchors = batch[4].cuda()
				labelgrids= rearrange(anchors, 'b c h w-> b c (h w)')
				labelgrids = rearrange(labelgrids, 'b c l-> (b c) l')
				labelgrids = torch.argmax(labelgrids, dim=1)
				optimizer.zero_grad()
				opState, opRot,prbmap, predicted_anchors = mpnet(input, labelops,labelrot, anchors)

				arcloss = wei_arc * smooLoss(opState, labelops)
				posloss = wei_pos * torch.sqrt(mseLoss(opState, labelops))
				rotloss = wei_rot * torch.sqrt(mseLoss(opRot, labelrot))
				holloss = wei_hol * holoLoss(opState,opRot[:,1:,:])
				unilos = wei_uni * uniLoss(opState,labelops)
				curloss = wei_cur * curLoss(opState, opRot)
				rsmloss = wei_rsm * rotsLoss(opRot, labelrot)
				safetyloss = wei_safety * safeLoss(opState, opRot, input)

				gridloss = wei_anchors *  crossentropyloss(predicted_anchors,labelgrids)
				loss = posloss + arcloss + holloss + unilos + rotloss + curloss + rsmloss + safetyloss+gridloss
				avg_posloss = avg_posloss + posloss.item()
				avg_arcloss = avg_arcloss + arcloss.item()
				avg_holloss = avg_holloss + holloss.item()
				avg_uniloss = avg_uniloss + unilos.item()
				avg_rotloss = avg_rotloss + rotloss.item()
				avg_curloss = avg_curloss + curloss.item()
				avg_rsmloss = avg_rsmloss + rsmloss.item()
				avg_safeloss = avg_safeloss + safetyloss.item()
				avg_anchorsloss = avg_anchorsloss + gridloss.item()
				avg_loss=avg_loss+loss.item()
				step_num += 1
				loss.backward()
				optimizer.step()
				pbar.update(input.shape[0])
				pbar.set_postfix(**{'loss (batch)': loss.item()})
				writer.add_scalar('mseloss', loss.item(), step_count)
				step_count += 1
    
				if(epoch>=4):
					compt_count+=1
					if compt_count%100 == 0:
						test_net(mpnet, val_loader, output_logfile2, wei_arc, smooLoss, wei_pos, mseLoss, wei_rot,
             wei_hol, holoLoss, wei_uni, uniLoss,wei_cur,curLoss, wei_rsm,rotsLoss,wei_safety,safeLoss,
             wei_anchors,crossentropyloss,compt_count)

		

		
		



		print ("--average loss: ", avg_loss/step_num)
		print ("--average arcloss: ", avg_arcloss/step_num)
		print ("--average holloss: ", avg_holloss/step_num)
		print ("--average uniloss: ", avg_uniloss/step_num)
		print ("--average posloss: ", avg_posloss/step_num)
		print ("--average rotloss: ", avg_rotloss/step_num)
		print ("--average curloss: ", avg_curloss/step_num)
		print ("--average rsmloss: ", avg_rsmloss/step_num)
		print ("--average safeoss: ", avg_safeloss/step_num)
		print ("--average anchors: ", avg_anchorsloss/step_num)
  
		output_logfile.write('--average loss: ' + str(avg_loss/step_num) + '\n')
		output_logfile.write('--average arcloss: ' + str(avg_arcloss/step_num) + '\n')
		output_logfile.write('--average holloss: ' + str(avg_holloss/step_num) + '\n')
		output_logfile.write('--average uniloss: ' + str(avg_uniloss/step_num) + '\n')
		output_logfile.write('--average posloss: ' + str(avg_posloss/step_num) + '\n')
		output_logfile.write('--average rotloss: ' + str(avg_rotloss/step_num) + '\n')
		output_logfile.write('--average curloss: ' + str(avg_curloss/step_num) + '\n')
		output_logfile.write('--average rsmloss: ' + str(avg_rsmloss/step_num) + '\n')
		output_logfile.write('--average safeloss: ' + str(avg_safeloss/step_num) + '\n')
		output_logfile.write('--average anchors: ' + str(avg_anchorsloss/step_num) + '\n')


		torch.save(mpnet.state_dict(),os.path.join(args.model_path,model_path))
	
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--model_path', type=str, default='./models/',help='path for saving trained models')
	# Model parameters
	parser.add_argument('--num_epochs', '-e', type=int, default=14)
	parser.add_argument('--batch_size','-b', type=int, default=32)
	parser.add_argument('--learning_rate','-l', type=float, default=1e-4)
	args = parser.parse_args()
	print(args)
	main(args)



