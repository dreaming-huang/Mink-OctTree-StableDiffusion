import torch
import MinkowskiEngine as ME
def test_union_map():                                                                                                                                                                                                                                                                 
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')                                                                                                                                                                                               
                                                                                                                                                                                                                                                                                        
        A = ME.SparseTensor(                                                                                                                                                                                                                                                              
                features=torch.FloatTensor([                                                                                                                                                                                                                                              
                    [1],                                                                                                                                                                                                                                                               
                    [1],                                                                                                                                                                                                                                                               
                    [1],                                                                                                                                                                                                                                                               
                    [1]                                                                                                                                                                                                                                                                
                ]),                                                                                                                                                                                                                                                                       
                coordinates=torch.IntTensor([                                                                                                                                                                                                                                             
                    [0, 0, 4],                                                                                                                                                                                                                                                         
                    [0, 0, 0],                                                                                                                                                                                                                                                         
                    [0, 0, 0],                                                                                                                                                                                                                                                         
                    [0, 4, 0]                                                                                                                                                                                                                                                          
                ]),                                                                                                                                                                                                                                                                       
                device=device,
                tensor_stride=4
                )                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
                                                                                                                                                                                                                                                                                       
        convtr_1 = ME.MinkowskiConvolutionTranspose(in_channels=1,                                                                                                                                                                                                                
                                                            out_channels=1,                                                                                                                                                                                                               
                                                            kernel_size=2,                                                                                                                                                                                                                
                                                            stride=2,                                                                                                                                                                                                                     
                                                            bias=False,                                                                                                                                                                                                                   
                                                            dimension=A.dimension).to(device)         
        convtr_2 = ME.MinkowskiGenerativeConvolutionTranspose(in_channels=1,                                                                                                                                                                                                                
                                                        out_channels=1,                                                                                                                                                                                                               
                                                        kernel_size=2,                                                                                                                                                                                                                
                                                        stride=2,                                                                                                                                                                                                                     
                                                        bias=False,                                                                                                                                                                                                                   
                                                        dimension=A.dimension).to(device)
        torch.nn.init.constant_(convtr_1.kernel, 1) 
        torch.nn.init.constant_(convtr_2.kernel, 1) 

        # print(A.dense()[0])                                                                                                                                                                                     
                                                                                                                                                                                                                                                                                        
        # # B = convtr_1(convtr_1(A)) 
        B = convtr_1(A)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
                                                                                                                                                                                                                                                                                        
        # print(B.C.shape)
        # # print(B.tensor_stride)
        # print(B.dense()[0])


        # # C = convtr_2(convtr_2(A))  
        # C = convtr_2(A)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
                                                                                                                                                                                                                                                                                        
        # print(C.C.shape)
        # # print(C.tensor_stride)
        # print(C.dense()[0])
        A.F=B.F
        print(A.F)


test_union_map()                                                                                                                                                                                                                                                                    