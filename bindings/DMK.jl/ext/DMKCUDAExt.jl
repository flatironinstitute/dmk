module DMKCUDAExt

using CUDA: CUDA
using DMK: DMK, DmkPtr, PdmkParams, DMK_MEMORY_DEVICE

Base.cconvert(::Type{DmkPtr}, x::CUDA.CuArray{T}) where {T} = Base.cconvert(CUDA.CuPtr{T}, x)
Base.unsafe_convert(::Type{DmkPtr}, x::CUDA.CuArray{T}) where {T} =
    reinterpret(DmkPtr, Base.unsafe_convert(CUDA.CuPtr{T}, x))

# DMK runs on its own streams, so the work queued on the task's stream has to finish first.
function DMK._check_array(params::PdmkParams, x::CUDA.CuArray)
    params.memory_space == DMK_MEMORY_DEVICE || error("device array passed with memory_space $(params.memory_space)")
    device = CUDA.deviceid(CUDA.device(x))
    device == params.gpu_device_id ||
        error("device array is on device $device, but gpu_device_id is $(params.gpu_device_id)")
    CUDA.synchronize()
    return nothing
end

end
