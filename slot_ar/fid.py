import torch_fidelity

metrics_dict = torch_fidelity.calculate_metrics(
    input1='/mnt/ceph_rbd/zbc/imagenet100_fid_test/', 
    input2='/mnt/ceph_rbd/zbc/imagenet100_ref/', 
    cuda=True, 
    isc=False, 
    fid=True, 
    kid=False, 
    prc=False, 
    verbose=True,
)
print(metrics_dict)

