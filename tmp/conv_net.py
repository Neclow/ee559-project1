import torch
from torch import nn
import math

class Net(nn.Module):
    def __init__(self,nConv_layers,in_conv=1,nLayers=2,kernel_size=3,kernel_pooling=2,multiple_conv=2,nParams=70000,multiple_fc=10,out_=10):
        super(Net, self).__init__()
        self.nConv_layers=nConv_layers
        self.in_conv=in_conv
        self.nLayers = nLayers
        self.kernel_size = kernel_size
        self.kernel_pooling = kernel_pooling if nConv_layers <=2 else 1
        self.multiple_conv=multiple_conv
        self.nParams = nParams
        self.multiple_fc=multiple_fc
        self.out_ = out_
        self.in_ = self.find_input_fc_size() 
        self.final_channel =0 
        print('found size ',  self.in_)
        
        self.out_channel=self.get_param_conv(nConv_layers,kernel_size,multiple_conv,multiple_fc,\
                                        nLayers, nParams, self.in_, out_)
        print('baseline number of channel :',self.out_channel)
        
        self.conv_layers =self.create_net_conv()
        self.fc_layers=self.create_net_fc()
        x=torch.ones((1,1,14,14))
        x=self.conv_layers(x)
        print('out_shape',x.shape)
        print('out_size= ',x.flatten().size(0))
        
        print(f'Parameters conv : {self.count_params_conv()}')
        print(f'Parameters fc : {self.count_params_fc()}')
        print(f'Parameters : {self.count_params()}')



        
        for p in self.parameters(): 
            print(p.shape)
            
    def create_net_conv(self,size=False):
        layers = []     
        
        
        if (size):
            out_channel = 1
                    
            for i in range(self.nConv_layers):
                layers.extend([nn.Conv2d(out_channel,out_channel,self.kernel_size),\
                           nn.MaxPool2d(kernel_size=(self.kernel_pooling,self.kernel_pooling)),\
                           nn.ReLU()])
        else:
        
            for i in range(1,self.nConv_layers+1):
                print('creating layer : ', i )
                out_channel=self.out_channel
                if i == 1 :
                    in_conv =self.in_conv
                    
                    layers.extend([nn.Conv2d(in_conv,out_channel,self.kernel_size),\
                               nn.MaxPool2d(kernel_size=(self.kernel_pooling)),\
                               nn.ReLU()])
                    in_conv =out_channel
                else:
                    
                    out_channel=(pow(self.multiple_conv,i-1)*self.out_channel)
                    print('exposant : ',(self.multiple_conv,i-1))
                    layers.extend([nn.Conv2d(in_conv,out_channel,self.kernel_size),\
                               nn.MaxPool2d(kernel_size=(self.kernel_pooling)),\
                               nn.ReLU()])
                    in_conv =out_channel
            self.final_channel = out_channel
                
            print('netwok created')
            
                
            
            
        return nn.Sequential(*layers)
    
            
    def create_net_fc(self):
        layers = []
        
        in_=round(self.in_*self.final_channel/(self.nConv_layers))
        print('in_',self.final_channel)
            
        for j in range (self.nLayers-1):
            layers.extend([nn.Linear(in_,\
                                         round(self.multiple_fc*self.out_channel)),nn.ReLU()])
            in_=round(self.multiple_fc*self.out_channel)
        
        # Output layer
        layers.append(nn.Linear(in_, self.out_))  
               
        return nn.Sequential(*layers) 
        
        
        
    def find_input_fc_size(self): 
        
        x=torch.ones((1,1,14,14))
        layers = self.create_net_conv(True)
        x = layers(x)
        return x.flatten().size(0)

    def forward(self, x):
        
        out = torch.zeros((x.shape[0], x.shape[1], self.out_))
        for i in range(x.shape[1]):
            
            conv=self.conv_layers(x[:,i,:,:])
            out[:,i,:] =self.create_net_fc(conv.view(-1, x_.flatten().size(0)))
      
        return out
        
    def count_params_conv(self):
        return sum(p.numel() for p in self.conv_layers.parameters()if p.requires_grad)
    def count_params_fc(self):
        return sum(p.numel() for p in self.fc_layers.parameters()if p.requires_grad)
    def count_params(self):
        return sum(p.numel() for p in self.parameters()if p.requires_grad)

    
    @staticmethod
    def get_param_conv(nConv_layers,kernel_size,multiple_conv,multiple_fc,*argv): 
        L, N, D, T = argv
        
        #param conv 
        x_1_=0
        for i in range(1,nConv_layers+1):
            if i==1: 
                x_1_ = (kernel_size**2+1) 
            else:
                x_1_ += pow(multiple_conv,(i-1))
        
        if(nConv_layers >1):
            x_2_=0
            j=1
            for i in range(2,nConv_layers+1):
                if i==2: 
                    interm_mult= ((multiple_conv),i-1)
                    x_2_ =multiple_conv*(kernel_size**2)


                    print('inter mult',interm_mult)
                else:  
                    interm_mult= (multiple_conv),((i-1)+(j))
                    x_2_ += pow((multiple_conv),((i-1)+(j)))*(kernel_size**2)
                    g=i-1+(j)
                    j = i-1

                    print('inter mult',interm_mult)
        else:
            x_2_=0
            
        #param_lin  
        y_1_=0
        y_2_=0
        f=0


        for i in range(1,L):
            if i == 1: 
                alpha = round(pow(multiple_conv,nConv_layers-1)/(nConv_layers))
                print('g',D*pow(multiple_conv,nConv_layers-1) )
                y_1_ += D*alpha
                y_2_ += D*alpha*multiple_fc
            else:   
                y_1_ += pow(multiple_fc,(i))
                y_2_ += pow(multiple_fc,(i-1)+i)
                f= i
           
        
        y_1_ += T*(pow(multiple_fc,f))
         
         
        a = y_2_+x_2_
        b= x_1_ + y_1_
        c = T - N
        s = pow(b**2-4*a*c, 0.5)
        res = ((-b+s)/(2*a))
        print('res',res)
        print('rouned',round(res))
        return round(res)