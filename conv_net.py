           

class Net(nn.Module):
    def __init__(self,nConv_layers,in_conv=2,nLayers=2,kernel_size=3,kernel_pooling=2,multiple_conv=5/4,nParams=70000,multiple_fc=1/3,out_=10):
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
        
        self.out_channel=self.get_param_conv(nConv_layers,kernel_size,multiple_conv,multiple_fc,\
                                        nLayers, nParams, self.in_, out_)
        print('baseline number of channel :',self.out_channel)
        
        self.conv_layers =self.create_net_conv()
        self.fc_layers=self.create_net_fc()
        
        print(f'Parameters: {self.count_params()}')

        
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
        
            for i in range(self.nConv_layers):
                print('creating layer : ', i )
                out_channel=self.out_channel
                if i == 0 :
                    in_conv =self.in_conv
                    
                    layers.extend([nn.Conv2d(in_conv,out_channel,self.kernel_size),\
                               nn.MaxPool2d(kernel_size=(self.kernel_pooling)),\
                               nn.ReLU()])
                    in_conv =out_channel
                else:
                    
                    out_channel=round(i*self.multiple_conv*self.out_channel)
                    layers.extend([nn.Conv2d(in_conv,out_channel,self.kernel_size),\
                               nn.MaxPool2d(kernel_size=(self.kernel_pooling)),\
                               nn.ReLU()])
                    in_conv =out_channel
                
            print('netwok created')
                
            
            
        return nn.Sequential(*layers)
    
            
    def create_net_fc(self):
        layers = []
        
        in_=self.in_
            
        for j in range (self.nLayers-1):
            layers.extend([nn.Linear(in_,round(self.multiple_fc*in_),nn.ReLU())])
            in_=round(self.multiple_fc*self.in_)
        
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
        
    def count_params(self):
        return sum(p.numel() for p in self.conv_layers.parameters())

    
    @staticmethod
    def get_param_conv(nConv_layers,kernel_size,multiple_conv,multiple_fc,*argv): 
        L, N, D, T = argv
        
        x_1_=0
        for i in range(1,nConv_layers+1):
            if i==1: 
                x_1_ = (kernel_size**2+1) 
            else:
                x_1_ +=(i-1)*multiple_conv
        
        if (nConv_layers >1):
            x_2_=0
            j=1
            for i in range(2,nConv_layers+1):
                if i==2: 
                    x_2_ =multiple_conv*(kernel_size**2)
                else:                
                    x_2_ += (multiple_conv**2)*(kernel_size**2)*(i-1)*(j)
                    j = i-1
        else:
            x_2_=0
            
        a =(L-1)*multiple_fc**2 + D*multiple_fc + x_2_
        b= x_1_ + (T+L)*multiple_fc
        c= T-N 
        s = pow(b**2-4*a*c, 0.5)
        return round((-b+s)/(2*a))