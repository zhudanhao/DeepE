import numpy as np
import math
import torch
import random
import copy
from scipy.stats import rankdata
import random

def evaluate(model,x_test,batch_size,target_dict):
    #target_dict:用于filter
    len_test = len(x_test)    
    batch_num = math.ceil(len(x_test) / batch_size)
    tail_scores_all = []
    tail_label = []
    
    for i in range(batch_num):
        batch_data  = x_test[batch_size*i:batch_size*(i+1)]
        batch_h,batch_r,batch_t = batch_data[:,0],batch_data[:,1],batch_data[:,2]
        tail_scores  = model.forward(batch_h,batch_r)
        tail_scores = tail_scores.cpu().detach().numpy()
        tail_scores_all.append(tail_scores)
        
        tail_label.append(batch_t)
        
    
    tail_scores_all = np.concatenate(tail_scores_all,axis=0)  
    
    tail_label = np.concatenate(tail_label,axis=0)  
    
    def cal_result(scores,labels,x_test,target_dict):
        ranks = []
        for i in range(len(labels)):
            arr = scores[i]
            mark = labels[i]
            h,r,t = x_test[i]
            mark_value = arr[mark]
            
            ##filter
            targets = target_dict[(h,r)]
            for target in targets:
                if target != mark:
                    arr[target] = np.finfo(np.float32).min
            ##
            rank = np.sum(arr>mark_value)
            rank+=1
            ranks.append(rank)
            
        mr, mrr, hits1, hits10 =0,[],[],[]
        mr = np.average(ranks)
        
        for rank in ranks:
            mrr.append(1/rank)
            if rank == 1:
                hits1.append(1)
            else:
                hits1.append(0)
            if rank <= 10:
                hits10.append(1)
            else:
                hits10.append(0)
        mrr = np.average(mrr)
        hits1 = np.average(hits1)
        hits10 = np.average(hits10)
        result = {'mr':mr, 'mrr':mrr, 'hits1':hits1, 'hits10':hits10}
        return result
    
    
    tail_result = cal_result(tail_scores_all,tail_label, x_test,target_dict)
    return {'mr':tail_result['mr'], 'mrr':tail_result['mrr'], 
            'hits@1':tail_result['hits1'], 'hits@10':tail_result['hits10']}

def better_than(re1,re2):
    if re1['mrr']>re2['mrr'] or re1['hits@10']>re2['hits@10']:
        return True
    else:
        return False


def train_epoch(train_doubles,num_batches_per_epoch,batch_size,model,opt,scheduler,x_valid,target_dict,num,device,max_mrr=0,epoch=1000,max_hits1=0,x_test=None):
    model.to(device)
    stop_num = 0
    previous_best = {'mr':-1,'mrr':-1,'hits@1':-1,'hits@10':-1,'epoch':-1}
    stop_start_epoch = int(0.4 * epoch)
    for epoch in range(epoch):
        model.train()
        random.shuffle(train_doubles) 
        losses = []
        for batch_num in range(num_batches_per_epoch):
            opt.zero_grad()
            x_batch = np.array(train_doubles[batch_num*batch_size:(batch_num+1)*batch_size])
            batch_h,batch_r,batch_t = x_batch[:,0],x_batch[:,1],x_batch[:,2]
            e1 = batch_h
            rel = batch_r
            e2_multi = batch_t
            
            pred = model.forward(e1, rel)

            loss = model.loss(pred, model.to_var(e2_multi))
            loss.backward()
            opt.step()
            losses.append(loss.detach().cpu().numpy())
        print(epoch,'train loss:',np.average(losses))
        scheduler.step(np.average(losses))
        model.eval()
        if epoch%10 == 0:
            print('valid',evaluate(model,x_valid,batch_size,target_dict))
            print('test',evaluate(model,x_test,batch_size,target_dict))
        with torch.no_grad():
            if epoch >= stop_start_epoch:
                    result = evaluate(model,x_valid,batch_size,target_dict)
                    if stop_num >= 40:
                        return best_model
                    if better_than(result,previous_best):
                        stop_num=0
                    else:
                        stop_num+=1  
                    if result['mrr'] > previous_best['mrr']:
                        previous_best = result
                        previous_best['epoch'] = epoch
                        best_model = copy.deepcopy(model)
    return best_model