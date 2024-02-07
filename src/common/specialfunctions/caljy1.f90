subroutine caljy1 ( arg, result, jint )

!*****************************************************************************80
!
!! CALJY1 computes various J1 and Y1 Bessel functions.
!
!  Discussion:
!
!    This routine computes first-order Bessel functions of the first and
!    second kind (J1 and Y1), for real arguments X, where 0 < X <= XMAX
!    for Y1, and |X| <= XMAX for J1.
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
!  Reference:
!
!    John Hart, Ward Cheney, Charles Lawson, Hans Maehly,
!    Charles Mesztenyi, John Rice, Henry Thatcher,
!    Christoph Witzgall,
!    Computer Approximations,
!    Wiley, 1968,
!    LC: QA297.C64.
!
!  Parameters:
!
!    Input, real ( kind = 8 ) ARG, the argument.  If JINT = 0, ARG
!    must satisfy
!     -XMAX < ARG < XMAX;
!    If JINT = 1, then ARG must satisfy
!      0 < ARG < XMAX.
!
!    Output, real ( kind = 8 ) RESULT, the value of the function,
!    which depends on the input value of JINT:
!    0, RESULT = J1(x);
!    1, RESULT = Y1(x);
!
!    Input, integer ( kind = 4 ) JINT, chooses the function to be computed.
!    0, J1(x);
!    1, Y1(x);
!
  implicit none

  real ( kind = 8 ) arg
  real ( kind = 8 ) ax
  real ( kind = 8 ) down
  real ( kind = 8 ) eight
  integer ( kind = 4 ) i
  integer ( kind = 4 ) jint
  real ( kind = 8 ) pi2
  real ( kind = 8 ) pj0(7)
  real ( kind = 8 ) pj1(8)
  real ( kind = 8 ) plg(4)
  real ( kind = 8 ) prod
  real ( kind = 8 ) py0(7)
  real ( kind = 8 ) py1(9)
  real ( kind = 8 ) p0(6)
  real ( kind = 8 ) p1(6)
  real ( kind = 8 ) p17
  real ( kind = 8 ) qj0(5)
  real ( kind = 8 ) qj1(7)
  real ( kind = 8 ) qlg(4)
  real ( kind = 8 ) qy0(6)
  real ( kind = 8 ) qy1(8)
  real ( kind = 8 ) q0(6)
  real ( kind = 8 ) q1(6)
  real ( kind = 8 ) resj
  real ( kind = 8 ) result
  real ( kind = 8 ) rtpi2
  real ( kind = 8 ) r0
  real ( kind = 8 ) r1
  real ( kind = 8 ) throv8
  real ( kind = 8 ) twopi
  real ( kind = 8 ) twopi1
  real ( kind = 8 ) twopi2
  real ( kind = 8 ) two56
  real ( kind = 8 ) up
  real ( kind = 8 ) w
  real ( kind = 8 ) wsq
  real ( kind = 8 ) xden
  real ( kind = 8 ) xinf
  real ( kind = 8 ) xmax
  real ( kind = 8 ) xnum
  real ( kind = 8 ) xsmall
  real ( kind = 8 ) xj0
  real ( kind = 8 ) xj1
  real ( kind = 8 ) xj01
  real ( kind = 8 ) xj02
  real ( kind = 8 ) xj11
  real ( kind = 8 ) xj12
  real ( kind = 8 ) xy
  real ( kind = 8 ) xy0
  real ( kind = 8 ) xy01
  real ( kind = 8 ) xy02
  real ( kind = 8 ) xy1
  real ( kind = 8 ) xy11
  real ( kind = 8 ) xy12
  real ( kind = 8 ) z
  real ( kind = 8 ) zsq
!
!  Mathematical constants
!
  data eight /8.0d0/
  data throv8 /0.375d0/
  data pi2 /6.3661977236758134308d-1/
  data p17 /1.716d-1/
  data twopi /6.2831853071795864769d+0/
  data twopi1 /6.28125d0/
  data twopi2 /1.9353071795864769253d-03/
  data two56 /256.0d+0/
  data rtpi2 /7.9788456080286535588d-1/
!
!  Machine-dependent constants
!
  data xmax /1.07d+09/
  data xsmall /9.31d-10/
  data xinf /1.7d+38/
!
!  Zeroes of Bessel functions
!
  data xj0 /3.8317059702075123156d+0/
  data xj1 /7.0155866698156187535d+0/
  data xy0 /2.1971413260310170351d+0/
  data xy1 /5.4296810407941351328d+0/
  data xj01 / 981.0d+0/
  data xj02 /-3.2527979248768438556d-04/
  data xj11 /1796.0d+0/
  data xj12 /-3.8330184381246462950d-05/
  data xy01 / 562.0d+0/
  data xy02 / 1.8288260310170351490d-03/
  data xy11 /1390.0d+0/
  data xy12 /-6.4592058648672279948d-06/
!
!  Coefficients for rational approximation to ln(x/a)
!
  data plg/-2.4562334077563243311d+01,2.3642701335621505212d+02, &
           -5.4989956895857911039d+02,3.5687548468071500413d+02/
  data qlg/-3.5553900764052419184d+01,1.9400230218539473193d+02, &
           -3.3442903192607538956d+02,1.7843774234035750207d+02/
!
!  Coefficients for rational approximation of
!  J1(X) / (X * (X**2 - XJ0**2)), XSMALL < |X| <=  4.0
!
  data pj0/9.8062904098958257677d+05,-1.1548696764841276794d+08, &
         6.6781041261492395835d+09,-1.4258509801366645672d+11, &
        -4.4615792982775076130d+03, 1.0650724020080236441d+01, &
        -1.0767857011487300348d-02/
  data qj0/5.9117614494174794095d+05, 2.0228375140097033958d+08, &
         4.2091902282580133541d+10, 4.1868604460820175290d+12, &
         1.0742272239517380498d+03/
!
!  Coefficients for rational approximation of
!  J1(X) / (X * (X**2 - XJ1**2)), 4.0 < |X| <= 8.0
!
  data pj1/4.6179191852758252280d+00,-7.1329006872560947377d+03, &
         4.5039658105749078904d+06,-1.4437717718363239107d+09, &
         2.3569285397217157313d+11,-1.6324168293282543629d+13, &
         1.1357022719979468624d+14, 1.0051899717115285432d+15/
  data qj1/1.1267125065029138050d+06, 6.4872502899596389593d+08, &
         2.7622777286244082666d+11, 8.4899346165481429307d+13, &
         1.7128800897135812012d+16, 1.7253905888447681194d+18, &
         1.3886978985861357615d+03/
!
!  Coefficients for rational approximation of
!  (Y1(X) - 2 LN(X/XY0) J1(X)) / (X**2 - XY0**2),
!  XSMALL < |X| <=  4.0
!
  data py0/2.2157953222280260820d+05,-5.9157479997408395984d+07, &
           7.2144548214502560419d+09,-3.7595974497819597599d+11, &
           5.4708611716525426053d+12, 4.0535726612579544093d+13, &
          -3.1714424660046133456d+02/
  data qy0/8.2079908168393867438d+02, 3.8136470753052572164d+05, &
           1.2250435122182963220d+08, 2.7800352738690585613d+10, &
           4.1272286200406461981d+12, 3.0737873921079286084d+14/
!
!  Coefficients for rational approximation of
!  (Y1(X) - 2 LN(X/XY1) J1(X)) / (X**2 - XY1**2),
!  4.0 < |X| <= 8.0
!
  data py1/ 1.9153806858264202986d+06,-1.1957961912070617006d+09, &
            3.7453673962438488783d+11,-5.9530713129741981618d+13, &
            4.0686275289804744814d+15,-2.3638408497043134724d+16, &
           -5.6808094574724204577d+18, 1.1514276357909013326d+19, &
           -1.2337180442012953128d+03/
  data qy1/ 1.2855164849321609336d+03, 1.0453748201934079734d+06, &
            6.3550318087088919566d+08, 3.0221766852960403645d+11, &
            1.1187010065856971027d+14, 3.0837179548112881950d+16, &
            5.6968198822857178911d+18, 5.3321844313316185697d+20/
!
!  Coefficients for Hart's approximation, 8.0 < |X|.
!
  data p0/-1.0982405543459346727d+05,-1.5235293511811373833d+06, &
           -6.6033732483649391093d+06,-9.9422465050776411957d+06, &
           -4.4357578167941278571d+06,-1.6116166443246101165d+03/
  data q0/-1.0726385991103820119d+05,-1.5118095066341608816d+06, &
           -6.5853394797230870728d+06,-9.9341243899345856590d+06, &
           -4.4357578167941278568d+06,-1.4550094401904961825d+03/
  data p1/ 1.7063754290207680021d+03, 1.8494262873223866797d+04, &
            6.6178836581270835179d+04, 8.5145160675335701966d+04, &
            3.3220913409857223519d+04, 3.5265133846636032186d+01/
  data q1/ 3.7890229745772202641d+04, 4.0029443582266975117d+05, &
            1.4194606696037208929d+06, 1.8194580422439972989d+06, &
            7.0871281941028743574d+05, 8.6383677696049909675d+02/
!
!  Check for error conditions.
!
  ax = abs ( arg )

  if ( jint == 1 .and. ( arg <= 0.0D+00 .or. &
    ( arg < 0.5D+00 .and. ax * xinf < pi2 ) ) ) then
    result = -xinf
    return
  else if ( xmax < ax ) then
    result = 0.0D+00
    return
  end if

  if ( eight < ax ) then
    go to 800
  else if ( ax <= xsmall ) then
    if ( jint == 0 ) then
      result = arg * 0.5D+00
    else
      result = -pi2 / ax
    end if
    return
  end if
!
!  Calculate J1 for appropriate interval, preserving
!  accuracy near the zero of J1.
!
  zsq = ax * ax

  if ( ax <= 4.0D+00 ) then
    xnum = ( pj0(7) * zsq + pj0(6) ) * zsq + pj0(5)
    xden = zsq + qj0(5)
    do i = 1, 4
      xnum = xnum * zsq + pj0(i)
      xden = xden * zsq + qj0(i)
    end do
    prod = arg * ( ( ax - xj01 / two56 ) - xj02 ) * ( ax + xj0 )
  else
    xnum = pj1(1)
    xden = ( zsq + qj1(7) ) * zsq + qj1(1)
    do i = 2, 6
      xnum = xnum * zsq + pj1(i)
      xden = xden * zsq + qj1(i)
    end do
    xnum = xnum * ( ax - eight ) * ( ax + eight ) + pj1(7)
    xnum = xnum * ( ax - 4.0D+00 ) * ( ax + 4.0D+00 ) + pj1(8)
    prod = arg * ( ( ax - xj11 / two56 ) - xj12 ) * ( ax + xj1 )
  end if

  result = prod * ( xnum / xden )

  if ( jint == 0 ) then
    return
  end if
!
!  Calculate Y1.  First find RESJ = pi/2 ln(x/xn) J1(x),
!  where xn is a zero of Y1.
!
  if ( ax <= 4.0D+00 ) then
    up = ( ax - xy01 / two56 ) - xy02
    xy = xy0
  else
    up = ( ax - xy11 / two56 ) - xy12
    xy = xy1
  end if

  down = ax + xy

  if ( abs ( up ) < p17 * down ) then
    w = up / down
    wsq = w * w
    xnum = plg(1)
    xden = wsq + qlg(1)
    do i = 2, 4
      xnum = xnum * wsq + plg(i)
      xden = xden * wsq + qlg(i)
    end do
    resj = pi2 * result * w * xnum / xden
  else
    resj = pi2 * result * log ( ax / xy )
  end if
!
!  Now calculate Y1 for appropriate interval, preserving
!  accuracy near the zero of Y1.
!
  if ( ax <= 4.0D+00 ) then
    xnum = py0(7) * zsq + py0(1)
    xden = zsq + qy0(1)
    do i = 2, 6
      xnum = xnum * zsq + py0(i)
      xden = xden * zsq + qy0(i)
    end do
  else
    xnum = py1(9) * zsq + py1(1)
    xden = zsq + qy1(1)
    do i = 2, 8
      xnum = xnum * zsq + py1(i)
      xden = xden * zsq + qy1(i)
    end do
  end if

  result = resj + ( up * down / ax ) * xnum / xden
  return
!
!  Calculate J1 or Y1 for 8.0 < |ARG|.
!
  800 continue

  z = eight / ax
  w = aint ( ax / twopi ) + throv8
  w = ( ax - w * twopi1 ) - w * twopi2
  zsq = z * z
  xnum = p0(6)
  xden = zsq + q0(6)
  up = p1(6)
  down = zsq + q1(6)

  do i = 1, 5
    xnum = xnum * zsq + p0(i)
    xden = xden * zsq + q0(i)
    up = up * zsq + p1(i)
    down = down * zsq + q1(i)
  end do

  r0 = xnum / xden
  r1 = up / down

  if ( jint == 0 ) then
    result = ( rtpi2 / sqrt ( ax ) ) &
      * ( r0 * cos ( w ) - z * r1 * sin ( w ) )
  else
    result = ( rtpi2 / sqrt ( ax ) ) &
      * ( r0 * sin ( w ) + z * r1 * cos ( w ) )
  end if

  if ( jint == 0 .and. arg < 0.0D+00 ) then
    result = -result
  end if

  return
end
