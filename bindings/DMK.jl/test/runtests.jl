using DMK
using Test

# `using DMK` above already ran the struct layout guard in DMK.__init__.

@testset "DMK" begin
    @testset "enums" begin
        @test Int(DMK_POTENTIAL) == 1
        @test Int(DMK_POTENTIAL_GRAD) == 2
        @test Int(DMK_VELOCITY) == 3
        @test Int(DMK_LAPLACE) == 1
        @test Int(DMK_EVAL_PATH_CPU) == 0
    end

    @testset "struct layout" begin
        @test sizeof(PdmkParams) == 136
        @test sizeof(PdmkEspParams) == 88
    end

    @testset "version" begin
        @test !isempty(pdmk_version_string())
        major, minor, patch = pdmk_version()
        @test (major, minor, patch) >= (0, 0, 0)
        @test !isempty(pdmk_git_commit())
    end

    @testset "laplace 3d against direct" begin
        n = 500
        r_src = rand(3n)
        charge = 2 .* rand(n) .- 1
        pot = zeros(n)
        reference = zeros(n)

        params = PdmkParams()
        params.n_dim = 3
        params.kernel = DMK_LAPLACE
        params.eps = 1e-9

        pdmk(C_NULL, params, n, r_src, charge, nothing, 0, nothing, pot, nothing)
        pdmk_direct(C_NULL, params, n, r_src, charge, nothing, 0, nothing, reference, nothing)

        @test sqrt(sum(abs2, pot .- reference) / sum(abs2, reference)) < 1e-7
    end

    @testset "required outputs" begin
        params = PdmkParams()
        params.n_dim = 3
        params.kernel = DMK_LAPLACE
        @test_throws ErrorException pdmk(C_NULL, params, 1, zeros(3), ones(1),
                                        nothing, 0, nothing, nothing, nothing)
    end
end
