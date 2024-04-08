export birdcageSensitivity
export measured2DSensitivity

"""
    birdcageSensitivity(N::Int64, ncoils::Int64, relative_radius::Float64)

Computes the sensitivity maps for each coils that are arranged
in a birdcage manner.
"""
function birdcageSensitivity(N::Int64,
                             ncoils::Int64,
                             relative_radius::Float64)

  out = zeros(ComplexF64, N, N, 1, ncoils)
  for c=0:ncoils-1
    coilx = relative_radius*cos(c*(2*pi/ncoils))
    coily = relative_radius*sin(c*(2*pi/ncoils))
    coil_phase = -c*(2*pi/ncoils)

    for y=0:N-1
      y_co = (y - (N/2))/(N/2) - coily
      for x=0:N-1
        x_co = (x - (N/2))/(N/2) - coilx
        rr = sqrt(x_co^2+y_co^2)
        phi = atan(x_co, -y_co) + coil_phase
        out[x+1,y+1, 1, c+1] = 1/(rr) * exp(1im*phi)
      end
    end
  end
 return out
end

"""
    measured2DSensitivity(N::Int64, ncoils::Int64)

Computes 2D coil sensitivity maps in image space
from actually measured coil profiles.
The code was originally implemented in the
Berkeley Advanced Reconstruction Toolbox (BART) by

  M. Uecker et al. Software toolbox and programming library for compressed sens-
  ing and parallel imaging. In: ISMRM Workshop on Data Sampling and
  Image Reconstruction, Sedona, 2013.

Checkout the original file:

  https://github.com/mrirecon/bart/blob/master/src/simu/sens.c

for more information.
"""

function get_sens_coeff(kx::Int,
                        ky::Int,
                        c::Int)

  sens_coeff = [[[ +2.202641e+02+7.959795e+02im,	+7.555631e+02-1.436550e+03im,	+2.545977e+03+6.890254e+02im,	-1.556657e+03+1.514510e+03im,	+2.119276e+02-1.494780e+03im ];;
  [ +1.298826e+03-2.217249e+02im,	+5.104937e+03-1.257278e+03im,	+2.718070e+03+9.347223e+03im,	-4.162577e+03+4.012845e+03im,	-2.934868e+02-1.494939e+02im ];;
  [ -2.586698e+03+3.241368e+02im,	+2.655842e+02-1.423245e+04im,	+3.444878e+04-5.959087e-07im,	+2.655845e+02+1.423245e+04im,	-2.586697e+03-3.241368e+02im ];;
  [ -2.934868e+02+1.494939e+02im,	-4.162577e+03-4.012845e+03im,	+2.718070e+03-9.347224e+03im,	+5.104937e+03+1.257278e+03im,	+1.298826e+03+2.217249e+02im ];;
  [ +2.119276e+02+1.494780e+03im,	-1.556657e+03-1.514511e+03im,	+2.545977e+03-6.890254e+02im,	+7.555631e+02+1.436550e+03im,	+2.202641e+02-7.959794e+02im ];;
  ];;;
  [ [ -5.140192e+02+3.919716e+02im,	+1.894585e+03-6.871808e+02im,	-1.588401e+03-9.072462e+02im,	+1.612868e+03+3.162505e+03im,	-1.764654e+03-7.521263e+01im ];;
  [ +1.468621e+03+2.462891e+03im,	+9.060663e+02-3.810722e+03im,	+1.295797e+04+1.274518e+03im,	+9.795861e+03+4.041410e+03im,	+5.382768e+03+1.496270e+03im ];;
  [ +6.432245e+03-1.873246e+03im,	-6.606911e+03+1.347433e+03im,	+1.469823e+04-2.828712e+04im,	+6.094570e+03-3.186246e+03im,	+1.553516e+03-6.658626e+03im ];;
  [ +1.552164e+02-2.608411e+03im,	-4.773840e+02+4.141277e+03im,	-1.697275e+04-4.176371e+03im,	+1.530776e+02-6.096529e+03im,	-1.256617e+03-3.806721e+03im ];;
  [ +1.427317e+03-6.593576e+02im,	+3.361672e+03+2.074898e+03im,	-1.207303e+03+5.915274e+03im,	-1.866361e+03+2.202192e+02im,	-2.244576e+03+5.222404e+02im ];;
  ];;;
  [ [ +1.501215e+03+5.783638e+02im,	-4.342622e+02-3.816680e+03im,	+1.776472e+03+3.912266e+03im,	+7.807321e+03-4.950209e+03im,	-2.595354e+03-2.460056e+03im ];;
  [ +2.369316e+03-2.313158e+03im,	+9.616589e+02-1.347896e+02im,	+1.163739e+04-1.087814e+04im,	+5.465111e+03-1.369369e+04im,	-2.497076e+03-7.660470e+03im ];;
  [ -1.154424e+03-5.152002e+03im,	+5.065459e+03+2.144815e+03im,	-8.206305e+03-2.679990e+04im,	-1.085187e+04-1.832581e+03im,	-1.138738e+04+2.371703e+03im ];;
  [ -1.613378e+03+1.783068e+03im,	+2.226849e+03-2.917332e+03im,	-1.126524e+04+1.535570e+04im,	-1.155414e+03+9.211537e+02im,	+2.441054e+03+3.475593e+03im ];;
  [ +2.149000e+03-1.050538e+03im,	+1.897796e+02-2.738874e+03im,	+9.831870e+03+3.711983e+01im,	+6.797585e+02+1.121489e+03im,	+2.977299e+03-2.676358e+02im ];;
  ];;;
  [ [ +9.225188e+02-1.609105e+03im,	+3.003085e+02-4.805967e+03im,	+3.010118e+03-5.193128e+03im,	-7.148681e+03-6.014095e+03im,	-5.118819e+03+6.886482e+03im ];;
  [ -7.304860e+01-1.906521e+03im,	+4.666944e+03-8.186537e+03im,	-4.628897e+03-1.670177e+04im,	-9.002204e+03-5.413904e+03im,	-5.991946e+03+3.112899e+03im ];;
  [ -1.707291e+03+1.371777e+03im,	+6.265779e+03-4.096543e+03im,	-1.623247e+04-1.249279e+04im,	-5.735404e+03+1.462694e+04im,	+3.235129e+03+7.745793e+03im ];;
  [ +2.658034e+03-2.003479e+03im,	-5.550209e+03-6.058204e+02im,	+4.231377e+03+1.547026e+04im,	+5.596798e+03-8.212248e+03im,	-2.146848e+02-8.379209e+03im ];;
  [ +7.590588e+02-2.001688e+03im,	+4.298658e+03-8.379312e+02im,	+8.961587e+02-1.114809e+04im,	-1.882075e+03+3.505513e+03im,	-9.920860e+01-1.958216e+01im ];;
  ];;;
  [ [ +1.038110e+03+1.053870e+03im,	+9.244695e+03+1.892697e+03im,	-3.192582e+03-6.490971e+03im,	-9.345550e+02-5.915621e+02im,	-5.048445e+03-5.094557e+02im ];;
  [ +6.029613e+03+9.528574e+03im,	+1.726803e+04+1.136792e+04im,	+1.704588e+04-6.327428e+03im,	-1.018518e+03-4.875073e+03im,	-1.385726e+03-2.624488e+03im ];;
  [ -5.607942e+03+3.239110e+03im,	-6.596133e+02+1.426120e+04im,	+1.907908e+04+6.801775e+03im,	-9.809665e+03-9.162689e+03im,	+2.646640e+03+3.444627e+03im ];;
  [ +3.773911e+03-3.830415e+03im,	-2.280376e+03-6.118903e+03im,	-9.791950e+03-1.582865e+03im,	+1.482043e+04+8.187234e+03im,	+2.503851e+03-5.622277e+03im ];;
  [ -3.532414e+02+2.797883e+03im,	+1.086923e+03+4.907002e+03im,	+2.711129e+03-4.658596e+02im,	-6.884295e+03+1.658655e+03im,	+2.195509e+03+1.532364e+03im ];;
  ];;;
  [ [ -2.660406e+03+1.130288e+03im,	-2.684663e+03-4.547482e+03im,	-1.091036e+03+1.039176e+04im,	+3.794462e+03-3.726057e+03im,	-9.599513e+02+2.510410e+03im ];;
  [ +2.453693e+03-1.127183e+04im,	+3.368645e+03-1.562005e+04im,	-6.638387e+03-2.965991e+03im,	+3.831320e+03+2.531652e+03im,	-2.198729e+03-5.958389e+02im ];;
  [ +1.428178e+04+9.163301e+02im,	+1.684055e+04-6.384269e+03im,	+1.658437e+04-1.232798e+04im,	-1.406222e+04+1.113432e+04im,	+3.592359e+03-6.771321e+03im ];;
  [ -1.660563e+03+2.839105e+03im,	-2.896235e+02+1.029472e+03im,	-3.908763e+03+3.005041e+03im,	+4.267630e+03-1.222684e+04im,	+3.162250e+03+3.059668e+03im ];;
  [ +1.229621e+03+1.214108e+03im,	+3.580651e+03+6.040717e+02im,	+7.638826e+02+2.616962e+03im,	+4.679096e+03+3.197829e+03im,	-2.025938e+03+1.366729e+03im ];;
  ];;;
  [ [ +1.532993e+03+3.085543e+03im,	+1.698160e+02-2.059932e+03im,	+8.620593e+03+5.947357e+03im,	-1.818730e+03-3.486291e+03im,	+2.804918e+03+2.657346e+02im ];;
  [ -5.363040e+03-8.659919e+03im,	-1.811624e+03-1.273996e+04im,	+5.336066e+02+4.722509e+03im,	+5.288265e+03-4.540653e+02im,	-1.550763e+03+3.172655e+03im ];;
  [ +1.484031e+04-8.435797e+03im,	+1.909462e+04-1.153338e+04im,	+2.644743e+04+1.609464e+03im,	-1.835121e+04+1.685986e+04im,	-2.060884e+03-1.164399e+04im ];;
  [ +3.565514e+03+4.812732e+03im,	-5.792477e+02-1.059542e+03im,	-6.948085e+03+1.922669e+03im,	-3.102303e+03-1.536498e+04im,	+8.313277e+03-1.413219e+03im ];;
  [ +3.645370e+03+3.144318e+03im,	+5.896468e+03-1.094847e+03im,	+3.137289e+03+4.010566e+03im,	+5.041110e+03-3.595689e+03im,	+4.239519e+03+4.264271e+03im ];;
  ];;;
  [ [ +9.757094e+02-4.944947e+02im,	-1.479025e+03-3.433955e+03im,	+6.554174e+03-3.580007e+03im,	-5.973520e+02+2.335976e+03im,	-1.082375e+03-1.958220e+03im ];;
  [ -2.523083e+03+1.680365e+03im,	-4.212411e+03-1.392797e+03im,	+2.731082e+03+4.483586e+03im,	-2.006126e+03-1.093595e+03im,	+1.536401e+03-1.071117e+03im ];;
  [ -9.129410e+03-5.654936e+03im,	-6.060043e+03-2.092658e+04im,	+2.299950e+04-2.185407e+04im,	+9.382429e+03+1.937377e+04im,	-1.130825e+04+2.193837e+03im ];;
  [ +6.269709e+02-2.393860e+03im,	-1.782260e+03-2.694656e+03im,	+2.224400e+03-3.487669e+01im,	-4.697312e+03+2.063353e+03im,	-1.975369e+03-3.797778e+03im ];;
  [ +8.142875e+02-3.201539e+03im,	+4.320247e+01-5.839946e+03im,	+7.607719e+03-2.151906e+03im,	-1.149005e+03+1.656691e+03im,	-3.673436e+01-2.858548e+03im ];;
  ];;]

	return sens_coeff[kx, ky, c]
end


function xsens(c::Int64,
               pos::Vector{Int64},
               dims::NTuple{4, Int64})

    @assert 8 >= c "Maximum number of supported coils was exceeded!"

    cpos = [(pos[1] - dims[1] / 2) / (0.5 * dims[1]), (pos[2] - dims[2] / 2) / (0.5 * dims[2])]
    val = 0 + 0im
    coil_coeff = 5
    shift = (coil_coeff - 1) / 2

    for i in 1:coil_coeff
        for j in 1:coil_coeff
            val += get_sens_coeff(i, j, c) * cis(-2 * π * ((i - shift) * cpos[1] + (j - shift) * cpos[2]) / 4)
        end
    end
    return val
end


function measured2DSensitivity(N::Int64,
                               ncoils::Int64)

    out = zeros(ComplexF32, N, N, 1, ncoils)

    for c in 1:ncoils, x in 1:N, y in 1:N
        out[x, y, 1, c] = xsens(c, [x, y], size(out))
    end
    return out
end