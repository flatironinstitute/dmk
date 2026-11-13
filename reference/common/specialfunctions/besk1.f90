function besk1 ( x )

!*****************************************************************************80
!
!! BESK1 evaluates the Bessel K1(X) function.
!
!  Discussion:
!
!    This routine computes approximate values for the
!    modified Bessel function of the second kind of order one
!    for arguments XLEAST <= ARG <= XMAX.
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
!    Output, real ( kind = 8 ) BESK1, the value of the function.
!
  implicit none

  real ( kind = 8 ) besk1
  integer ( kind = 4 ) jint
  real ( kind = 8 ) result
  real ( kind = 8 ) x

  jint = 1
  call calck1 ( x, result, jint )
  besk1 = result

  return
end
