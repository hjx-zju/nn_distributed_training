import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim
def MSE(pred, gt):
    
    error=np.mean((pred-gt)**2)
    return error
def SSIM(pred, gt):
    
    return ssim(gt, pred, data_range=gt.max() - gt.min())
def PSNR(pred, gt):
    mse = MSE(pred, gt)
    return 10 * np.log10(1 / mse)
#Baron’s cross correlation coefficient
def BCCC(pred, gt):
    pred = pred.flatten()
    gt = gt.flatten()
    pred_mean = np.mean(pred)
    gt_mean = np.mean(gt)
    pred_std = np.std(pred)
    gt_std = np.std(gt)
    n = len(pred)
    bccc = np.sum((pred-pred_mean)*(gt-gt_mean))/(n*pred_std*gt_std)
    return bccc
def print_value(pred,gt,name,occupied=0.8,empty=0.2):
    
    if type(pred) == torch.Tensor:
            pred = pred.numpy()
    print(name,"MSE: ", MSE(pred, gt),"SSIM: ", SSIM(pred, gt),"PSNR: ", PSNR(pred, gt),"BCCC: ", BCCC(pred, gt))
    return
   
def FalsePositive(pred, gt,occupied=0.8,empty=0.2):
   
    false_positive = np.sum((pred >occupied) & (gt <empty))/np.sum(gt<empty)
    return false_positive
def FalseNegative(pred, gt,occupied=0.8,empty=0.2):
    false_negative = np.sum((pred <empty) & (gt >occupied))/np.sum(gt >occupied)
    return false_negative

# from utils.metric import ImageSimilarityMetric
# score,list_score=ImageSimilarityMetric(pred3,gt,lidar.img[::8, ::8].shape)
# def ImageSimilarityMetric(pred, gt,shape):
#     score=0
#     list_score=[]
#     pred=pred.reshape(shape)
#     gt=gt.reshape(shape)
#     print(gt.shape)
#     occupied, occluded, free = computeSimilarityMetric(gt,pred)
#     score += occupied
#     score += occluded
#     score += free
#     list_score.append(occupied)
#     list_score.append(occluded)
#     list_score.append(free)

   

#     return score,list_score

# def toDiscrete(m):
#     """
#     Args:
#         - m (m,n) : np.array with the occupancy grid
#     Returns:
#         - discrete_m : thresholded m
#     """
#     # print(m.shape)
#     y_size, x_size = m.shape
#     m_occupied = np.zeros(m.shape)
#     m_free = np.zeros(m.shape)
#     m_occluded = np.zeros(m.shape)

#     #Handpicked
#     occupied_value = 0.8
#     occluded_value = 0.2

#     m_occupied[m >= occupied_value] = 1.0
#     m_occluded[np.logical_and(m >= occluded_value, m < occupied_value)] = 1.0
#     m_free[m < occluded_value] = 1.0

#     return m_occupied, m_occluded, m_free

# def todMap(m):

#     """
#     Extra if statements are for edge cases.
#     """


#     y_size, x_size = m.shape
#     dMap = np.ones(m.shape) * np.Inf

#     dMap[m == 1] = 0.0

#     for y in range(0,y_size):
#         if y == 0:
#             for x in range(1,x_size):
#                 h = dMap[y,x-1]+1
#                 dMap[y,x] = min(dMap[y,x], h)

#         else:
#             for x in range(0,x_size):
#                 if x == 0:
#                     h = dMap[y-1,x]+1
#                     dMap[y,x] = min(dMap[y,x], h)
#                 else:
#                     h = min(dMap[y,x-1]+1, dMap[y-1,x]+1)
#                     dMap[y,x] = min(dMap[y,x], h)

#     for y in range(y_size-1,-1,-1):

#         if y == y_size-1:
#             for x in range(x_size-2,-1,-1):
#                 h = dMap[y,x+1]+1
#                 dMap[y,x] = min(dMap[y,x], h)

#         else:
#             for x in range(x_size-1,-1,-1):
#                 if x == x_size-1:
#                     h = dMap[y+1,x]+1
#                     dMap[y,x] = min(dMap[y,x], h)
#                 else:
#                     h = min(dMap[y+1,x]+1, dMap[y,x+1]+1)
#                     dMap[y,x] = min(dMap[y,x], h)

#     return dMap

# def computeDistance(m1,m2):

#     y_size, x_size = m1.shape
#     dMap = todMap(m2)
#     # d = 0
#     # num_cells = 0
#     d = np.sum(dMap[m1 == 1])
#     num_cells = np.sum(m1 == 1)

#     if num_cells != 0:
#         output = d/num_cells

#     if num_cells == 0 or d == np.Inf:
#         output = y_size + x_size

#     return output

# def computeSimilarityMetric(m1, m2):

#     m1_occupied, m1_occluded, m1_free = toDiscrete(m1)
#     m2_occupied, m2_occluded, m2_free = toDiscrete(m2)

#     occluded = computeDistance(m2_occluded, m1_occluded) + computeDistance(m1_occluded, m2_occluded)
#     occupied = computeDistance(m1_occupied,m2_occupied) + computeDistance(m2_occupied,m1_occupied)
#     free = computeDistance(m1_free,m2_free) + computeDistance(m2_free,m1_free)

#     return occupied, occluded, free
