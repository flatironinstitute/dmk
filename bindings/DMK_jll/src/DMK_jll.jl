# Use baremodule to shave off a few KB from the serialized `.ji` file
baremodule DMK_jll
using Base
using Base: UUID
using LazyArtifacts
using MPIPreferences
Base.include(@__MODULE__, joinpath("..", ".pkg", "platform_augmentation.jl"))
import JLLWrappers

JLLWrappers.@generate_main_file_header("DMK")
JLLWrappers.@generate_main_file("DMK", Base.UUID("dd75985e-f840-5a39-a51a-8eefcb4c69b4"))
end  # module DMK_jll
