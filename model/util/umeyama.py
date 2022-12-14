"""
Changed to torch version.
"""
import numpy as np
import torch
import cv2
import itertools

def estimateSimilarityTransform(source, target, verbose=False):
    device = source.device
    dtype = source.dtype
    SourceHom = torch.cat([source, torch.ones((source.shape[0], 1), device=device, dtype=dtype)], dim=1).transpose(0, 1)
    TargetHom = torch.cat([target, torch.ones((source.shape[0], 1), device=device, dtype=dtype)], dim=1).transpose(0, 1)

    # Auto-parameter selection based on source-target heuristics
    TargetNorm = torch.mean(torch.linalg.norm(target, dim=1))
    SourceNorm = torch.mean(torch.linalg.norm(source, dim=1))
    RatioTS = (TargetNorm / SourceNorm)
    RatioST = (SourceNorm / TargetNorm)
    PassT = RatioST if(RatioST>RatioTS) else RatioTS
    StopT = PassT / 100
    nIter = 100
    if verbose:
        print('Pass threshold: ', PassT)
        print('Stop threshold: ', StopT)
        print('Number of iterations: ', nIter)

    SourceInliersHom, TargetInliersHom, BestInlierRatio = getRANSACInliers(SourceHom, TargetHom, MaxIterations=nIter, PassThreshold=PassT, StopThreshold=StopT)
    if(BestInlierRatio < 0.1):
        print('[ WARN ] - Something is wrong. Small BestInlierRatio: ', BestInlierRatio)
        return None, None, None, None

    Scales, Rotation, Translation, OutTransform = estimateSimilarityUmeyama(SourceInliersHom, TargetInliersHom)

    if verbose:
        print('BestInlierRatio:', BestInlierRatio)
        print('Rotation:\n', Rotation)
        print('Translation:\n', Translation)
        print('Scales:', Scales)

    return Scales, Rotation, Translation, OutTransform

def estimateRestrictedAffineTransform(source: np.array, target: np.array, verbose=False):
    raise NotImplementedError
    SourceHom = np.transpose(np.hstack([source, np.ones([source.shape[0], 1])]))
    TargetHom = np.transpose(np.hstack([target, np.ones([source.shape[0], 1])]))

    RetVal, AffineTrans, Inliers = cv2.estimateAffine3D(source, target)
    # We assume no shear in the affine matrix and decompose into rotation, non-uniform scales, and translation
    Translation = AffineTrans[:3, 3]
    NUScaleRotMat = AffineTrans[:3, :3]
    # NUScaleRotMat should be the matrix SR, where S is a diagonal scale matrix and R is the rotation matrix (equivalently RS)
    # Let us do the SVD of NUScaleRotMat to obtain R1*S*R2 and then R = R1 * R2
    R1, ScalesSorted, R2 = np.linalg.svd(NUScaleRotMat, full_matrices=True)

    if verbose:
        print('-----------------------------------------------------------------------')
    # Now, the scales are sort in ascending order which is painful because we don't know the x, y, z scales
    # Let's figure that out by evaluating all 6 possible permutations of the scales
    ScalePermutations = list(itertools.permutations(ScalesSorted))
    MinResidual = 1e8
    Scales = ScalePermutations[0]
    OutTransform = np.identity(4)
    Rotation = np.identity(3)
    for ScaleCand in ScalePermutations:
        CurrScale = np.asarray(ScaleCand)
        CurrTransform = np.identity(4)
        CurrRotation = (np.diag(1 / CurrScale) @ NUScaleRotMat).transpose()
        CurrTransform[:3, :3] = np.diag(CurrScale) @ CurrRotation
        CurrTransform[:3, 3] = Translation
        # Residual = evaluateModel(CurrTransform, SourceHom, TargetHom)
        Residual = evaluateModelNonHom(source, target, CurrScale,CurrRotation, Translation)
        if verbose:
            # print('CurrTransform:\n', CurrTransform)
            print('CurrScale:', CurrScale)
            print('Residual:', Residual)
            print('AltRes:', evaluateModelNoThresh(CurrTransform, SourceHom, TargetHom))
        if Residual < MinResidual:
            MinResidual = Residual
            Scales = CurrScale
            Rotation = CurrRotation
            OutTransform = CurrTransform

    if verbose:
        print('Best Scale:', Scales)

    if verbose:
        print('Affine Scales:', Scales)
        print('Affine Translation:', Translation)
        print('Affine Rotation:\n', Rotation)
        print('-----------------------------------------------------------------------')

    return Scales, Rotation, Translation, OutTransform

def getRANSACInliers(SourceHom, TargetHom, MaxIterations=100, PassThreshold=200, StopThreshold=1):
    device = SourceHom.device
    dtype = SourceHom.dtype
    BestResidual = 1e10
    BestInlierRatio = 0
    BestInlierIdx = torch.arange(SourceHom.shape[1], device=device, dtype=dtype)
    for i in range(0, MaxIterations):
        # Pick 5 random (but corresponding) points from source and target
        RandIdx = torch.randint(0, SourceHom.shape[1], (5,))
        _, _, _, OutTransform = estimateSimilarityUmeyama(SourceHom[:, RandIdx], TargetHom[:, RandIdx])
        Residual, InlierRatio, InlierIdx = evaluateModel(OutTransform, SourceHom, TargetHom, PassThreshold)
        if Residual < BestResidual:
            BestResidual = Residual
            BestInlierRatio = InlierRatio
            BestInlierIdx = InlierIdx
        if BestResidual < StopThreshold:
            break

        # print('Iteration: ', i)
        # print('Residual: ', Residual)
        # print('Inlier ratio: ', InlierRatio)

    return SourceHom[:, BestInlierIdx], TargetHom[:, BestInlierIdx], BestInlierRatio

def evaluateModel(OutTransform, SourceHom, TargetHom, PassThreshold):
    Diff = TargetHom - torch.matmul(OutTransform, SourceHom)
    ResidualVec = torch.linalg.norm(Diff[:3, :], dim=0)
    Residual = torch.linalg.norm(ResidualVec)
    InlierIdx = (ResidualVec < PassThreshold).nonzero().reshape(-1)
    nInliers = InlierIdx.shape[0]
    InlierRatio = nInliers / SourceHom.shape[1]
    return Residual, InlierRatio, InlierIdx

def evaluateModelNoThresh(OutTransform, SourceHom, TargetHom):
    raise NotImplementedError
    Diff = TargetHom - np.matmul(OutTransform, SourceHom)
    ResidualVec = np.linalg.norm(Diff[:3, :], axis=0)
    Residual = np.linalg.norm(ResidualVec)
    return Residual

def evaluateModelNonHom(source, target, Scales, Rotation, Translation):
    raise NotImplementedError
    RepTrans = np.tile(Translation, (source.shape[0], 1))
    TransSource = (np.diag(Scales) @ Rotation @ source.transpose() + RepTrans.transpose()).transpose()
    Diff = target - TransSource
    ResidualVec = np.linalg.norm(Diff, axis=0)
    Residual = np.linalg.norm(ResidualVec)
    return Residual

def testNonUniformScale(SourceHom, TargetHom):
    raise NotImplementedError
    OutTransform = np.matmul(TargetHom, np.linalg.pinv(SourceHom))
    ScaledRotation = OutTransform[:3, :3]
    Translation = OutTransform[:3, 3]
    Sx = np.linalg.norm(ScaledRotation[0, :])
    Sy = np.linalg.norm(ScaledRotation[1, :])
    Sz = np.linalg.norm(ScaledRotation[2, :])
    Rotation = np.vstack([ScaledRotation[0, :] / Sx, ScaledRotation[1, :] / Sy, ScaledRotation[2, :] / Sz])
    print('Rotation matrix norm:', np.linalg.norm(Rotation))
    Scales = np.array([Sx, Sy, Sz])

    # # Check
    # Diff = TargetHom - np.matmul(OutTransform, SourceHom)
    # Residual = np.linalg.norm(Diff[:3, :], axis=0)
    return Scales, Rotation, Translation, OutTransform

def estimateSimilarityUmeyama(SourceHom, TargetHom):
    # Copy of original paper is at: http://web.stanford.edu/class/cs273/refs/umeyama.pdf
    device = SourceHom.device
    dtype = SourceHom.dtype
    SourceCentroid = torch.mean(SourceHom[:3, :], dim=1)
    TargetCentroid = torch.mean(TargetHom[:3, :], dim=1)
    nPoints = SourceHom.shape[1]

    CenteredSource = SourceHom[:3, :] - SourceCentroid[:,None].repeat(1, nPoints)
    CenteredTarget = TargetHom[:3, :] - TargetCentroid[:,None].repeat(1, nPoints)

    CovMatrix = torch.matmul(CenteredTarget, CenteredSource.transpose(0,1)) / nPoints

    if torch.isnan(CovMatrix).any():
        print('nPoints:', nPoints)
        print(SourceHom.shape)
        print(TargetHom.shape)
        raise RuntimeError('There are NANs in the input.')

    U, D, Vh = torch.linalg.svd(CovMatrix, full_matrices=True)
    d = (torch.linalg.det(U) * torch.linalg.det(Vh)) < 0.0
    if d:
        D[-1] = -D[-1]
        U[:, -1] = -U[:, -1]

    Rotation = torch.matmul(U, Vh).T # Transpose is the one that works

    varP = torch.var(SourceHom[:3, :], dim=1).sum()
    ScaleFact = 1/varP * torch.sum(D) # scale factor
    Scales = ScaleFact.reshape(-1).repeat(3)
    ScaleMatrix = torch.diag(Scales)
    Translation = TargetHom[:3, :].mean(dim=1) - SourceHom[:3, :].mean(dim=1)[None].mm(ScaleFact*Rotation)

    OutTransform = torch.eye(4, device=device, dtype=dtype)
    OutTransform[:3, :3] = torch.matmul(ScaleMatrix, Rotation)
    OutTransform[:3, 3] = Translation

    # # Check
    # Diff = TargetHom - np.matmul(OutTransform, SourceHom)
    # Residual = np.linalg.norm(Diff[:3, :], axis=0)

    return Scales, Rotation, Translation, OutTransform