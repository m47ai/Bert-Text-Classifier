import torch

def train(epoch,model,training_loader,device,optimizer,loss_fn):
    
    model.train()
    
    for _,batch in enumerate(training_loader, 0):
        
        ids = batch['input_ids'].to(device, dtype = torch.long)
        mask = batch['attention_mask'].to(device, dtype = torch.long)
        token_type_ids = batch['token_type_ids'].to(device, dtype = torch.long)
        targets = batch['targets'].to(device, dtype = torch.float)

        outputs = model(ids, mask, token_type_ids)

        optimizer.zero_grad()

        loss = loss_fn(outputs, targets)

        if _%5000==0:

            print(f'Epoch: {epoch}, Loss:  {loss.item()}')
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def validation(epoch,model,testing_loader,device,optimizer,loss_fn):

    model.eval()
    
    fin_targets=[]
    
    fin_outputs=[]
    
    with torch.no_grad():
    
        for _, batch in enumerate(testing_loader, 0):
    
            ids = batch['input_ids'].to(device, dtype = torch.long)
    
            mask = batch['attention_mask'].to(device, dtype = torch.long)
    
            token_type_ids = batch['token_type_ids'].to(device, dtype = torch.long)
    
            targets = batch['targets'].to(device, dtype = torch.float)
    
            outputs = model(ids, mask, token_type_ids)
    
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
    
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
    
    return fin_outputs, fin_targets