export reconstruction_simple, reconstruction_multiEcho, reconstruction_multiCoil, reconstruction_multiCoilMultiEcho, reconstruction_lowRank, RecoParameters

"""
Performs iterative image reconstruction independently for the data of all coils,
contrasts and slices

# Arguments
* `acqData::AcquisitionData`            - AcquisitionData object
* `reconSize::NTuple{2,Int64}`              - size of image to reconstruct
* `reg::Regularization`                 - Regularization to be used
* `sparseTrafo::AbstractLinearOperator` - sparsifying transformation
* `weights::Vector{Vector{Complex{<:AbstractFloat}}}` - sampling density of the trajectories in acqData
* `solvername::String`                  - name of the solver to use
* (`normalize::Bool=false`)             - adjust regularization parameter according to the size of k-space data
* (`params::Dict{Symbol,Any}`)          - Dict with additional parameters
"""
function reconstruction_simple( acqData::AcquisitionData{T}
                              , reconSize::NTuple{D,Int64}
                              , reg::Vector{Regularization}
                              , sparseTrafo
                              , weights::Vector{Vector{Complex{T}}}
                              , solvername::String
                              , normalize::Bool=false
                              , encodingOps=nothing
                              , params::Dict{Symbol,Any}=Dict{Symbol,Any}()) where {D, T <: AbstractFloat}

  encDims = ndims(trajectory(acqData))
  if encDims!=length(reconSize)
    error("reco-dimensionality $(length(reconSize)) and encoding-dimensionality $(encDims) do not match")
  end

  numContr, numChan, numSl, numRep = numContrasts(acqData), numChannels(acqData), numSlices(acqData), numRepetitions(acqData)

  encParams = getEncodingOperatorParams(;params...)

  # set sparse trafo in reg
  reg[1].params[:sparseTrafo] = sparseTrafo

  # reconstruction
  Ireco = zeros(Complex{T}, prod(reconSize), numSl, numContr, numChan, numRep)
  #@floop
  for l = 1:numRep, k = 1:numSl
    if encodingOps!=nothing
      F = encodingOps[:,k]
    else
      F = encodingOps_simple(acqData, reconSize, slice=k; encParams...)
    end
    for j = 1:numContr
      W = WeightingOp(weights[j])
      for i = 1:numChan
        kdata = kData(acqData,j,i,k,rep=l).* weights[j]
        EFull = ∘(W, F[j])#, isWeighting=true)
        EFullᴴEFull = normalOperator(EFull)
        solver = createLinearSolver(solvername, EFull; AᴴA=EFullᴴEFull, reg=reg, params...)

        I = solve(solver, kdata, startVector=get(params,:startVector,Complex{T}[]),
                              solverInfo=get(params,:solverInfo,nothing))

        if isCircular( trajectory(acqData, j) )
          circularShutter!(reshape(I, reconSize), 1.0)
        end
        Ireco[:,k,j,i,l] = I
      end
    end
  end
  Ireco = reshape(Ireco, volumeSize(reconSize, numSl)..., numContr, numChan, numRep)

  return makeAxisArray(Ireco, acqData)
end

"""
Performs a iterative image reconstruction jointly for all contrasts. Different slices and coil images
are reconstructed independently.

# Arguments
* `acqData::AcquisitionData`            - AcquisitionData object
* `reconSize::NTuple{2,Int64}`              - size of image to reconstruct
* `reg::Regularization`                 - Regularization to be used
* `sparseTrafo::AbstractLinearOperator` - sparsifying transformation
* `weights::Vector{Vector{Complex{<:AbstractFloat}}}` - sampling density of the trajectories in acqData
* `solvername::String`                  - name of the solver to use
* (`normalize::Bool=false`)             - adjust regularization parameter according to the size of k-space data
* (`params::Dict{Symbol,Any}`)          - Dict with additional parameters
"""
function reconstruction_multiEcho(acqData::AcquisitionData{Complex{T}}
                              , reconSize::NTuple{D,Int64}
                              , reg::Vector{Regularization}
                              , sparseTrafo
                              , weights::Vector{Vector{Complex{T}}}
                              , solvername::String
                              , normalize::Bool=false
                              , encodingOps=nothing
                              , params::Dict{Symbol,Any}=Dict{Symbol,Any}()) where {D , T <: AbstractFloat}

  encDims = ndims(trajectory(acqData))
  if encDims!=length(reconSize)
    error("reco-dimensionality $(length(reconSize)) and encoding-dimensionality $(encDims) do not match")
  end

  numContr, numChan, numSl, numRep = numContrasts(acqData), numChannels(acqData), numSlices(acqData), numRepetitions(acqData)
  encParams = getEncodingOperatorParams(;params...)

  # set sparse trafo in reg
  reg[1].params[:sparseTrafo] = DiagOp( repeat([sparseTrafo],numContr)... )

  W = WeightingOp( vcat(weights...) )

  # reconstruction
  Ireco = zeros(Complex{T}, prod(reconSize)*numContr, numChan, numSl, numRep)
  #@floop for l = 1:numRep, i = 1:numSl
  for l = 1:numRep, i = 1:numSl
    if encodingOps != nothing
      F = encodingOps[i]
    else
      F = encodingOp_multiEcho(acqData, reconSize, slice=i; encParams...)
    end
    for j = 1:numChan
      kdata = multiEchoData(acqData, j, i,rep=l) .* vcat(weights...)
      EFull = ∘(W, F[j])#, isWeighting=true)
      EFullᴴEFull = normalOperator(EFull)
      solver = createLinearSolver(solvername, EFull; AᴴA=EFullᴴEFull, reg=reg, params...)

      Ireco[:,j,i,l] = solve(solver,kdata; params...)
      # TODO circular shutter
    end
  end

  if encDims==2
    # 2d reconstruction
    Ireco = reshape(Ireco, reconSize[1], reconSize[2], numContr, numChan, numSl,numRep)
    Ireco = permutedims(Ireco, [1,2,5,3,4,6])
  else
    # 3d reconstruction
    Ireco = reshape(Ireco, reconSize[1], reconSize[2], reconSize[3], numContr, numChan,numRep)
  end

  return makeAxisArray(permutedims(Ireco,[1,2,5,3,4,6]), acqData)
end

"""
Performs a SENSE-type iterative image reconstruction. Different slices and contrasts images
are reconstructed independently.

# Arguments
* `acqData::AcquisitionData`            - AcquisitionData object
* `reconSize::NTuple{2,Int64}`              - size of image to reconstruct
* `reg::Regularization`                 - Regularization to be used
* `sparseTrafo::AbstractLinearOperator` - sparsifying transformation
* `weights::Vector{Vector{Complex{<:AbstractFloat}}}` - sampling density of the trajectories in acqData
* `L_inv::Array{Complex{<:AbstractFloat}}`        - noise decorrelation matrix
* `solvername::String`                  - name of the solver to use
* `senseMaps::Array{Complex{<:AbstractFloat}}`        - coil sensitivities
* (`normalize::Bool=false`)             - adjust regularization parameter according to the size of k-space data
* (`params::Dict{Symbol,Any}`)          - Dict with additional parameters
"""
function reconstruction_multiCoil(acqData::AcquisitionData{T}
                              , reconSize::NTuple{D,Int64}
                              , reg::Vector{Regularization}
                              , sparseTrafo
                              , weights::Vector{Vector{Complex{T}}}
                              , L_inv::Union{LowerTriangular{Complex{T}, Matrix{Complex{T}}}, Nothing}
                              , solvername::String
                              , senseMaps::Array{Complex{T}}
                              , normalize::Bool=false
                              , encodingOps=nothing
                              , params::Dict{Symbol,Any}=Dict{Symbol,Any}()) where {D , T}

  encDims = ndims(trajectory(acqData))
  if encDims!=length(reconSize)
    error("reco-dimensionality $(length(reconSize)) and encoding-dimensionality $(encDims) do not match")
  end

  numContr, numChan, numSl, numRep = numContrasts(acqData), numChannels(acqData), numSlices(acqData), numRepetitions(acqData)
  encParams = getEncodingOperatorParams(;params...)

  # noise decorrelation
  senseMapsUnCorr = decorrelateSenseMaps(L_inv, senseMaps, numChan)

  # set sparse trafo in reg
  reg[1].params[:sparseTrafo] = sparseTrafo

  # solve optimization problem
  Ireco = zeros(Complex{T}, prod(reconSize), numSl, numContr, numRep)
  # @floop for l = 1:numRep, k = 1:numSl
  for l = 1:numRep, k = 1:numSl
    if encodingOps != nothing
      E = encodingOps[:,k]
    else
      E = encodingOps_parallel(acqData, reconSize, senseMapsUnCorr; slice=k, encParams...)
    end

    for j = 1:numContr
      W = WeightingOp(weights[j],numChan)
      kdata = multiCoilData(acqData, j, k, rep=l) .* repeat(weights[j], numChan)
      if !isnothing(L_inv)
        kdata = vec(reshape(kdata, :, numChan) * L_inv')
      end

      EFull = ∘(W, E[j], isWeighting=true)
      EFullᴴEFull = normalOperator(EFull)
      solver = createLinearSolver(solvername, EFull; AᴴA=EFullᴴEFull, reg=reg, params...)
      I = solve(solver, kdata; params...)

      if isCircular( trajectory(acqData, j) )
        circularShutter!(reshape(I, reconSize), 1.0)
      end
      Ireco[:,k,j,l] = I
    end
  end

  Ireco_ = reshape(Ireco, volumeSize(reconSize, numSl)..., numContr, 1,numRep)

  return makeAxisArray(Ireco_, acqData)
end


"""
Performs a SENSE-type iterative image reconstruction which reconstructs all contrasts jointly.
Different slices are reconstructed independently.

# Arguments
* `acqData::AcquisitionData`            - AcquisitionData object
* `reconSize::NTuple{2,Int64}`              - size of image to reconstruct
* `reg::Regularization`                 - Regularization to be used
* `sparseTrafo::AbstractLinearOperator` - sparsifying transformation
* `weights::Vector{Vector{Complex{<:AbstractFloat}}}` - sampling density of the trajectories in acqData
* `solvername::String`                  - name of the solver to use
* `senseMaps::Array{Complex{<:AbstractFloat}}`        - coil sensitivities
* (`normalize::Bool=false`)             - adjust regularization parameter according to the size of k-space data
* (`params::Dict{Symbol,Any}`)          - Dict with additional parameters
"""
function reconstruction_multiCoilMultiEcho(acqData::AcquisitionData{T}
                              , reconSize::NTuple{D,Int64}
                              , reg::Vector{Regularization}
                              , sparseTrafo
                              , weights::Vector{Vector{Complex{T}}}
                              , L_inv::Union{LowerTriangular{Complex{T}, Matrix{Complex{T}}}, Nothing}
                              , solvername::String
                              , senseMaps::Array{Complex{T}}
                              , normalize::Bool=false
                              , encodingOps=nothing
                              , params::Dict{Symbol,Any}=Dict{Symbol,Any}()) where {D, T}

  encDims = ndims(trajectory(acqData))
  if encDims!=length(reconSize)
    error("reco-dimensionality $(length(reconSize)) and encoding-dimensionality $(encDims) do not match")
  end

  numContr, numChan, numSl, numRep = numContrasts(acqData), numChannels(acqData), numSlices(acqData), numRepetitions(acqData)
  encParams = getEncodingOperatorParams(;params...)

  # noise decorrelation
  senseMapsUnCorr = decorrelateSenseMaps(L_inv, senseMaps, numChan)

  # set sparse trafo in reg
  reg[1].params[:sparseTrafo] = DiagOp( repeat([sparseTrafo],numContr)... )

  W = WeightingOp( vcat(weights...), numChan )

  Ireco = zeros(Complex{T}, prod(reconSize)*numContr, numSl, numRep)
  # @floop for l = 1:numRep, i = 1:numSl
  for l = 1:numRep, i = 1:numSl
    if encodingOps != nothing
      E = encodingOps[i]
    else
      E = encodingOp_multiEcho_parallel(acqData, reconSize, senseMapsUnCorr; slice=i, encParams...)
    end

    kdata = multiCoilMultiEchoData(acqData, i) .* repeat(vcat(weights...), numChan)
    if !isnothing(L_inv)
      kdata = vec(reshape(kdata, :, numChan) * L_inv')
    end

    EFull = ∘(W, E)#, isWeighting=true)
    EFullᴴEFull = normalOperator(EFull)
    solver = createLinearSolver(solvername, EFull; AᴴA=EFullᴴEFull, reg=reg, params...)

    Ireco[:,i,l] = solve(solver, kdata; params...)
  end


  if encDims==2
    # 2d reconstruction
    Ireco = reshape(Ireco, reconSize[1], reconSize[2], numContr, numSl, 1, numRep)
    Ireco = permutedims(Ireco, [1,2,4,3,5,6])
  else
    # 3d reconstruction
    Ireco = reshape(Ireco, reconSize[1], reconSize[2], reconSize[3], numContr, 1, numRep)
  end

  return makeAxisArray(Ireco, acqData)
end

"""
Performs a SENSE-type iterative image reconstruction for multi-interleave spiral images with phase correction.
Different slices and contrasts images are reconstructed independently.

# Arguments
* `acqData::AcquisitionData`            - AcquisitionData object that includes all spiral interleaves
* `reconSize::NTuple{2,Int64}`          - size of image to reconstruct
* `reg::Regularization`                 - Regularization to be used
* `sparseTrafo::AbstractLinearOperator` - sparsifying transformation
* `weights::Vector{Vector{Complex{<:AbstractFloat}}}` - sampling density of the trajectories in acqData
* `L_inv::Array{Complex{<:AbstractFloat}}`        - noise decorrelation matrix
* `solvername::String`                  - name of the solver to use
* `senseMaps::Array{Complex{<:AbstractFloat}}`        - coil sensitivities
* (`normalize::Bool=false`)             - adjust regularization parameter according to the size of k-space data
* (`params::Dict{Symbol,Any}`)          - Dict with additional parameters
"""
function reconstruction_multiInterleave(acqData::AcquisitionData{T}
                              , reconSize::NTuple{D,Int64}
                              , reg::Vector{Regularization}
                              , sparseTrafo
                              , weights::Vector{Vector{Complex{T}}}
                              , L_inv::Union{LowerTriangular{Complex{T}, Matrix{Complex{T}}}, Nothing}
                              , solvername::String
                              , senseMaps::Array{Complex{T}}
                              , normalize::Bool=false
                              , encodingOps=nothing
                              , params::Dict{Symbol,Any}=Dict{Symbol,Any}()) where {D , T}

  encDims = ndims(trajectory(acqData))
  if encDims!=length(reconSize)
    error("reco-dimensionality $(length(reconSize)) and encoding-dimensionality $(encDims) do not match")
  end

  numContr, numChan, numSl, numRep = numContrasts(acqData), numChannels(acqData), numSlices(acqData), numRepetitions(acqData)
  encParams = getEncodingOperatorParams(;params...)

  # noise decorrelation
  senseMapsUnCorr = decorrelateSenseMaps(L_inv, senseMaps, numChan)

  # set sparse trafo in reg
  reg[1].params[:sparseTrafo] = sparseTrafo

  # Split interleaves into multiple AcquisitionData objects
  acqDataSet = splitMultiInterleaves(acqData)
  numInterleave = length(acqDataSet)
  pcMaps = params[:intlvPhaseMaps] # phase correction maps.
  numSampPerRO = acqData.traj[1].numSamplingPerProfile

  # solve optimization problem
  Ireco = zeros(Complex{T}, prod(reconSize), numSl, numContr, numRep)
  # @floop for l = 1:numRep, k = 1:numSl
  for l = 1:numRep, k = 1:numSl
    if encodingOps != nothing
      E = encodingOps[:,k]
    else      
      E = Vector{Vector{MRIOperators.CompositeOp}}(undef, numInterleave)
      for t = 1 : numInterleave
        senseMapTemp = senseMapsUnCorr .* repeat(pcMaps[t], 1, 1, 1, numChan)
        E[t] = encodingOps_parallel(acqDataSet[t], reconSize, senseMapTemp; slice=k, encParams...)
      end
    end

    for j = 1:numContr
      # Pre-weighting of k-space data
      kdata = multiCoilData(acqData, j, k, rep=l) .* repeat(weights[j], numChan)

      # Reshape k-space data, let spiral interleaves to be the last dimension.
      kdata = reshape(kdata, (numSampPerRO, numInterleave, numChan))
      kdata = permutedims(kdata, (1, 3, 2))
      kdata = vec(kdata)
      
      # A vector to store full operator E for each spiral interleave
      EAllIntlv = Vector{MRIOperators.CompositeOp}(undef, numInterleave)

      # Calculate each full encoding matrix (including density compensation) for each spiral interleave/shot
      for t = 1 : numInterleave
        idxStart = (t-1) * numSampPerRO + 1 # Start and end index in kdata and traj of the current interleave
        idxEnd = t * numSampPerRO
        WTemp = WeightingOp(weights[j][idxStart : idxEnd],numChan)
        EAllIntlv[t] = ∘(deepcopy(WTemp), E[t][j], isWeighting=true)   
      end
      
      # Concatenate full encoding matrices vertically since interleave/shot is the last dimension of k-space data.
      EFull = deepcopy(EAllIntlv[1])
      for t = 2 : numInterleave
        EFull = vcat(EFull, EAllIntlv[t])
      end

      EFullᴴEFull = normalOperator(EFull)
      
      solver = createLinearSolver(solvername, EFull; AᴴA=EFullᴴEFull, reg=reg, params...)
      I = solve(solver, kdata; params...)

      if isCircular( trajectory(acqData, j) )
        circularShutter!(reshape(I, reconSize), 1.0)
      end
      Ireco[:,k,j,l] = I
    end
  end

  Ireco_ = reshape(Ireco, volumeSize(reconSize, numSl)..., numContr, 1,numRep)

  return makeAxisArray(Ireco_, acqData)
end


"""
    splitMultiInterleaves(AcqData::AcquisitionData)
Split an AcquisitionData with multiple interleaves into multiple AcquisitionData with single interleave.
Returns an array with multiple AcquisitionData that each includes single interleave.
This is the same as the function in GIRFReco, but here serves as an auxilary function for `reconstruction_multiInterleave`

# Arguments
* `AcqData`          - An AcquisitionData with multiple interleaves
"""
function splitMultiInterleaves(acqData::AcquisitionData)
    numInterleave = acqData.traj[1].numProfiles
    numSampPerRO = acqData.traj[1].numSamplingPerProfile

    result = []

    for l = 1 : numInterleave
        # Index of readout in traj, times, kdata
        idxStart = (l-1) * numSampPerRO + 1
        idxEnd = l * numSampPerRO

        # Copy the original acqData
        acqDataTemp = deepcopy(acqData)

        # Extract and correct the trajectory
        acqDataTemp.traj[1].numProfiles = 1
        acqDataTemp.traj[1].nodes = acqDataTemp.traj[1].nodes[:, idxStart:idxEnd]
        acqDataTemp.traj[1].times = acqDataTemp.traj[1].times[idxStart:idxEnd]

        # Extract the kspace data
        for m = 1 : length(acqData.kdata)
            acqDataTemp.kdata[m] = acqDataTemp.kdata[m][idxStart:idxEnd, :]
        end

        # Correct the sub sampling indices
        acqDataTemp.subsampleIndices[1] = 1 : numSampPerRO

        push!(result, acqDataTemp)
    end

    return result
end