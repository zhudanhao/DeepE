import numpy as np
import math
import torch
import random
import copy
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




def train_epoch(train_doubles,num_batches_per_epoch,batch_size,model,opt,scheduler,x_test,target_dict,num,device,max_mrr=0,epoch=1000,max_hits1=0):
    model.to(device)
    stop_num=0
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

            loss = model.loss(pred, model.to_var(e2_multi)) #+ 1e-0* model.l2_reg_loss()
            loss.backward()
            opt.step()
            losses.append(loss.detach().cpu().numpy())
        print(epoch,'train loss:',np.average(losses))
        scheduler.step(np.average(losses))
        model.eval()
        with torch.no_grad():
                #if epoch % 5 == 0 and epoch > 0:
                    #print(evaluate(model,x_valid,batch_size,target_dict))
            
            if epoch % num == 0 and epoch > 0:
                if epoch > 0:
                    print(evaluate(model,x_test,batch_size,target_dict))
                    if evaluate(model,x_test,batch_size,target_dict)['mrr'] > max_mrr :
                        max_mrr = evaluate(model,x_test,batch_size,target_dict)['mrr']
                        best_model = copy.deepcopy(model)
                        #stop_num=0
                    #elif evaluate(model,x_test,batch_size,target_dict)['hits@1'] > max_hits1:
                        # max_hits1 = evaluate(model,x_test,batch_size,target_dict)['hits@1']
                        # stop_num=0
                    # else:
                        # stop_num+=1
            # if stop_num == 200:
                # break
                           
    return best_model