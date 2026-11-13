function besj1 ( x )

!*****************************************************************************80
!
!! BESJ1 evaluates the Bessel J1(X) function.
!
!  Discussion:
!
!    This routine computes approximate values for Bessel functions
!    of the first kind of order zero for arguments  |X| <= XMAX
!
!    See comments heading CALJY1.
!
!  Licensing:
!
!    This code is distributed under the GNU LGPL license.
!
!  Modified:
!
!    03 April 2007
!
!  Author:
!
!    Original FORTRAN77 version by William Cody.
!    FORTRAN90 version by John Burkardt.
!
!  Parameters:
!
!    Input, real ( kind = 8 ) X, the argument of the function.
!
!    Output, real ( kind = 8 ) BESJ1, the value of the function.
!
  implicit none

  real ( kind = 8 ) besj1
  integer ( kind = 4 ) jint
  real ( kind = 8 ) result
  real ( kind = 8 ) x

  jint = 0
  call caljy1 ( x, result, jint )
  besj1 = result

  return
end
