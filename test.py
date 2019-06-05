import torch
import torchvision

import os
from CNNArchitecture import Net

# device
device = "cpu"

# data loader
root = "/home/santiago/dataset/Market-1501-v15.09.15/pytorch/"
query_dir = os.path.join(root,"query")
gallery_dir = os.path.join(root,"gallery")
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((128,64)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
queryloader = torch.utils.data.DataLoader(
    torchvision.datasets.ImageFolder(query_dir, transform=transform),
    batch_size=64, shuffle=False
)
galleryloader = torch.utils.data.DataLoader(
    torchvision.datasets.ImageFolder(gallery_dir, transform=transform),
    batch_size=64, shuffle=False
)

# net definition
net = Net(get_features=True,num_classes=751)##mars 625 ,market1501 751
assert os.path.isfile("./checkpoint/ckpt.pytorch"), "Error: no checkpoint file found!"
print('Loading from checkpoint/ckpt.pytorch')
checkpoint = torch.load("./checkpoint/ckpt.pytorch")



net_dict = checkpoint['net_dict']
net.load_state_dict(net_dict)
net.eval()
net.to(device)

# compute features
query_features = torch.tensor([]).float()
query_labels = torch.tensor([]).long()
gallery_features = torch.tensor([]).float()
gallery_labels = torch.tensor([]).long()




# =============================================================================
# torch.autograd.no_grad
#     Context-manager that disabled gradient calculation.
# 
#     Disabling gradient calculation is useful for inference, when you are sure that you will not call Tensor.backward().
#     It will reduce memory consumption for computations that would otherwise have requires_grad=True. In this mode, 
#     the result of every computation will have requires_grad=False, even when the inputs have requires_grad=True.
# =============================================================================


with torch.no_grad():
    for idx,(inputs,labels) in enumerate(queryloader):
        inputs = inputs.to(device)
        features = net(inputs).cpu()
        query_features = torch.cat((query_features, features), dim=0)
        query_labels = torch.cat((query_labels, labels))
        print("queryloader ",idx)
        #if(idx>=10): break

    for idx,(inputs,labels) in enumerate(galleryloader):
        inputs = inputs.to(device)
        features = net(inputs).cpu()
        gallery_features = torch.cat((gallery_features, features), dim=0)
        gallery_labels = torch.cat((gallery_labels, labels))
        print("galleryloader ",idx)
        #if(idx>=10): break

gallery_labels -= 2

# =============================================================================
#  torch.cat(tensors, dim=0, out=None) → Tensor
# 
#     Concatenates the given sequence of seq tensors in the given dimension. 
#All tensors must either have the same shape (except in the concatenating dimension) or be empty.
# 
#     torch.cat() can be seen as an inverse operation for torch.split() and torch.chunk().
# 
#     torch.cat() can be best understood via examples.
# 
#     Parameters
# 
#             tensors (sequence of Tensors) – any python sequence of tensors of the same type. 
#Non-empty tensors provided must have the same shape, except in the cat dimension.
# 
#             dim (int, optional) – the dimension over which the tensors are concatenated
# 
#             out (Tensor, optional) – the output tensor
# =============================================================================





# save features
features = {
    "qf": query_features,
    "ql": query_labels,
    "gf": gallery_features,
    "gl": gallery_labels
}
print(features)

qf = features["qf"]
ql = features["ql"]
gf = features["gf"]
gl = features["gl"]
print(gf)
scores = qf.mm(gf.t())
# =============================================================================
#  torch.mm(mat1, mat2, out=None) → Tensor
# 
#     Performs a matrix multiplication of the matrices mat1 and mat2.
# 
#     If mat1 is a (n×m) tensor, mat2 is a (m×p) tensor, out will be a (n×p) tensor.
# =============================================================================
    
res = scores.topk(5, dim=1)[1][:,0]


# =============================================================================
#  torch.topk(input, k, dim=None, largest=True, sorted=True, out=None) -> (Tensor, LongTensor)
# 
#     Returns the k largest elements of the given input tensor along a given dimension.
# 
#     If dim is not given, the last dimension of the input is chosen.
# 
#     If largest is False then the k smallest elements are returned.
# 
#     A namedtuple of (values, indices) is returned, where the indices are the indices of the elements in the original input tensor.
# 
#     The boolean option sorted if True, will make sure that the returned k elements are themselves sorted
# 
#     Parameters
# 
#             input (Tensor) – the input tensor
# 
#             k (int) – the k in “top-k”
# 
#             dim (int, optional) – the dimension to sort along
# 
#             largest (bool, optional) – controls whether to return largest or smallest elements
# 
#             sorted (bool, optional) – controls whether to return the elements in sorted order
# 
#             out (tuple, optional) – the output tuple of (Tensor, LongTensor) that can be optionally given to be used as output buffers
# =============================================================================


top1correct = gl[res].eq(ql).sum().item()
# =============================================================================
#  torch.eq(input, other, out=None) → Tensor
# 
#     Computes element-wise equality
# 
#     The second argument can be a number or a tensor whose shape is broadcastable with the first argument.
# =============================================================================

print("Acc top1:{:.3f}".format(top1correct/ql.size(0)))






