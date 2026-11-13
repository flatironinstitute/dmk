      subroutine bdmk_pwterms(eps,ilev,npw)
C     Get planewave expansion length for many deltas at the same level ilev
c      
      implicit real*8 (a-h,o-z)

      if (ilev.lt.-10) goto 2000
      if (eps.gt.1d-3 .and. eps.le.1d-2) then
         call dmkpwterms_md2(ilev,npw)
      elseif (eps.gt.1d-4 .and. eps.le.1d-3) then
         call dmkpwterms_md3(ilev,npw)
      elseif (eps.gt.1d-5 .and. eps.le.1d-4) then
         call dmkpwterms_md4(ilev,npw)
      elseif (eps.gt.1d-6 .and. eps.le.1d-5) then
         call dmkpwterms_md5(ilev,npw)
      elseif (eps.gt.1d-7 .and. eps.le.1d-6) then
         call dmkpwterms_md6(ilev,npw)
      elseif (eps.gt.1d-8 .and. eps.le.1d-7) then
         call dmkpwterms_md7(ilev,npw)
      elseif (eps.gt.1d-9 .and. eps.le.1d-8) then
         call dmkpwterms_md8(ilev,npw)
      elseif (eps.gt.1d-10 .and. eps.le.1d-9) then
         call dmkpwterms_md9(ilev,npw)
      elseif (eps.gt.1d-11 .and. eps.le.1d-10) then
         call dmkpwterms_md10(ilev,npw)
      elseif (eps.gt.1d-12 .and. eps.le.1d-11) then
         call dmkpwterms_md11(ilev,npw)
      endif
      return
      
 2000 continue
      pi=4*atan(1.0d0)
      npw=2*ceiling(2*log(10/eps)/pi)
      return
      end 
c
c
c
c
      subroutine dmkpwterms_md2(ilev,npw)
C     Get planewave expansion length for many deltas at the same level ilev
c      
      implicit real*8 (a-h,o-z)
      integer npw0(-10:0)
      data npw0/2,2,2,2,2,2,2,3,4,6,8/ 

      if (ilev .le. 0) then
         npw=npw0(ilev)*2
      else
         npw=npw0(0)*2
      endif
      
      return
      end 
c
c
c
c
      subroutine dmkpwterms_md3(ilev,npw)
C     Get planewave expansion length for many deltas at the same level ilev
c      
      implicit real*8 (a-h,o-z)
      integer npw0(-10:0)
      data npw0/3,3,3,3,3,3,3,4,5,8,11/

      if (ilev .le. 0) then
         npw=npw0(ilev)*2
      else
         npw=npw0(0)*2
      endif
      
      return
      end 
c
c
c
c
      subroutine dmkpwterms_md4(ilev,npw)
C     Get planewave expansion length for many deltas at the same level ilev
c      
      implicit real*8 (a-h,o-z)
      integer npw0(-10:0)
      data npw0/4,4,4,4,3,4,4,5,7,10,15/

      if (ilev .le. 0) then
         npw=npw0(ilev)*2
      else
         npw=npw0(0)*2
      endif
      
      return
      end 
c
c
c
c
      subroutine dmkpwterms_md5(ilev,npw)
C     Get planewave expansion length for many deltas at the same level ilev
c      
      implicit real*8 (a-h,o-z)
      integer npw0(-10:0)
      data npw0/4,4,4,4,4,4,5,6,8,12,19/
      
      if (ilev .le. 0) then
         npw=npw0(ilev)*2
      else
         npw=npw0(0)*2
      endif
      
      return
      end 
c
c
c
c
      subroutine dmkpwterms_md6(ilev,npw)
C     Get planewave expansion length for many deltas at the same level ilev
c      
      implicit real*8 (a-h,o-z)
      integer npw0(-10:0)
      data npw0/5,5,5,5,5,5,6,7,9,14,22/

      if (ilev .le. 0) then
         npw=npw0(ilev)*2
      else
         npw=npw0(0)*2
      endif
      
      return
      end 
c
c
c
c
      subroutine dmkpwterms_md7(ilev,npw)
C     Get planewave expansion length for many deltas at the same level ilev
c      
      implicit real*8 (a-h,o-z)
      integer npw0(-10:0)
      data npw0/5,6,6,6,5,6,7,8,11,16,26/


      if (ilev .le. 0) then
         npw=npw0(ilev)*2
      else
         npw=npw0(0)*2
      endif
      
      return
      end 
c
c
c
c
      subroutine dmkpwterms_md8(ilev,npw)
C     Get planewave expansion length for many deltas at the same level ilev
c      
      implicit real*8 (a-h,o-z)
      integer npw0(-10:0)
      data npw0/6,6,6,6,6,7,7,9,12,18,29/

      if (ilev .le. 0) then
         npw=npw0(ilev)*2
      else
         npw=npw0(0)*2
      endif
      
      return
      end 
c
c
c
c
      subroutine dmkpwterms_md9(ilev,npw)
C     Get planewave expansion length for many deltas at the same level ilev
c      
      implicit real*8 (a-h,o-z)
      integer npw0(-10:0)
      data npw0/7,7,7,7,7,7,8,10,13,20,33/

      if (ilev .le. 0) then
         npw=npw0(ilev)*2
      else
         npw=npw0(0)*2
      endif
      
      return
      end 
c
c
c
c
      subroutine dmkpwterms_md10(ilev,npw)
C     Get planewave expansion length for many deltas at the same level ilev
c      
      implicit real*8 (a-h,o-z)
      integer npw0(-10:0)
      data npw0/7,7,8,8,8,8,9,11,15,22,36/

      if (ilev .le. 0) then
         npw=npw0(ilev)*2
      else
         npw=npw0(0)*2
      endif
      
      return
      end 
c
c
c
c
      subroutine dmkpwterms_md11(ilev,npw)
C     Get planewave expansion length for many deltas at the same level ilev
c      
      implicit real*8 (a-h,o-z)
      integer npw0(-10:0)
      data npw0/8,8,8,8,8,9,10,12,16,25,40/


      if (ilev .le. 0) then
         npw=npw0(ilev)*2
      else
         npw=npw0(0)*2
      endif
      
      return
      end 
c     
c
c
c
c*********************************************************************
C
C get plane wave approximation nodes and weights
C
C*********************************************************************
      subroutine get_pwnodes_md(eps,nlevels,ilev,npw,ws,ts,bs0)
C
C     Get planewave expansion weights,nodes for many deltas at the same level ilev
c      
      implicit real *8 (a-h,o-z)
      real *8 ws(*),ts(*)

      if (ilev.lt.-10) goto 2000
      if (eps.gt.1d-3 .and. eps.le.1d-2) then
         call get_pwnodes_md2(nlevels,ilev,npw,ws,ts)
      elseif (eps.gt.1d-4 .and. eps.le.1d-3) then
         call get_pwnodes_md3(nlevels,ilev,npw,ws,ts)
      elseif (eps.gt.1d-5 .and. eps.le.1d-4) then
         call get_pwnodes_md4(nlevels,ilev,npw,ws,ts)
      elseif (eps.gt.1d-6 .and. eps.le.1d-5) then
         call get_pwnodes_md5(nlevels,ilev,npw,ws,ts)
      elseif (eps.gt.1d-7 .and. eps.le.1d-6) then
         call get_pwnodes_md6(nlevels,ilev,npw,ws,ts)
      elseif (eps.gt.1d-8 .and. eps.le.1d-7) then
         call get_pwnodes_md7(nlevels,ilev,npw,ws,ts)
      elseif (eps.gt.1d-9 .and. eps.le.1d-8) then
         call get_pwnodes_md8(nlevels,ilev,npw,ws,ts)
      elseif (eps.gt.1d-10 .and. eps.le.1d-9) then
         call get_pwnodes_md9(nlevels,ilev,npw,ws,ts)
      elseif (eps.gt.1d-11 .and. eps.le.1d-10) then
         call get_pwnodes_md10(nlevels,ilev,npw,ws,ts)
      elseif (eps.gt.1d-12 .and. eps.le.1d-11) then
         call get_pwnodes_md11(nlevels,ilev,npw,ws,ts)
      endif

      do i=1,npw
         ts(i)=ts(i)/bs0
         ws(i)=ws(i)/bs0
      enddo

      return
      
 2000 continue
      deltamax=4.00**(-ilev)/log(10/eps)
      pi=4*atan(1.0d0)
      h=2*pi/sqrt(deltamax*log(10/eps))
      
cccc      print *, ilev, deltamax, h
      npw2=npw/2
      do i=1,npw2
         ts(i+npw2)=h*(i-0.5d0)
         ws(i+npw2)=h/2/sqrt(pi)
      enddo

      do i=1,npw2
         ts(npw2-i+1)=-ts(i+npw2)
         ws(npw2-i+1)=ws(i+npw2)
      enddo

      do i=1,npw
         ts(i)=ts(i)/bs0
         ws(i)=ws(i)/bs0
      enddo
      
      return
      end 
c
c
c
c
      subroutine get_pwnodes_md2(nlevels,ilev,npw,ws,ts)
C
C     Get planewave expansion weights,nodes for many deltas at the same level ilev
c     2 digits of accuracy
c      
      implicit real *8 (a-h,o-z)
      real *8 ws(*),ts(*)
      real *8 wsm10(2),tsm10(2)
      real *8 wsm9(2),tsm9(2)
      real *8 wsm8(2),tsm8(2)
      real *8 wsm7(2),tsm7(2)
      real *8 wsm6(2),tsm6(2)
      real *8 wsm5(2),tsm5(2)
      real *8 wsm4(2),tsm4(2)
      real *8 wsm3(3),tsm3(3)
      real *8 wsm2(4),tsm2(4)
      real *8 wsm1(6),tsm1(6)
      real *8 ws0(8),ts0(8)
c        
c  Data for   2 nodes
c        
c        Nodes:
c        
      data tsm10/
     1  0.2958734893481696D-02,0.9789491740881303D-02/
c        
c        Weights:
c        
      data wsm10/
     1  0.6044603198176865D-02,0.8147568528877734D-02/
c        
c        
c  Data for   2 nodes
c        
c        Nodes:
c        
      data tsm9/
     1  0.5917227263306967D-02,0.1957801113855034D-01/
c        
c        Weights:
c        
      data wsm9/
     1  0.1208868783992019D-01,0.1629406462879535D-01/
c        
c        
c  Data for   2 nodes
c        
c        Nodes:
c        
      data tsm8/
     1  0.1183252726841346D-01,0.3914829732732158D-01/
c        
c        Weights:
c        
      data wsm8/
     1  0.2417325569901357D-01,0.3257960237759942D-01/
c        
c        
c  Data for   2 nodes
c        
c        Nodes:
c        
      data tsm7/
     1  0.2365005994699296D-01,0.7823655447689443D-01/
c        
c        Weights:
c        
      data wsm7/
     1  0.4831448359166908D-01,0.6509270693620972D-01/
c        
c        
c  Data for   2 nodes
c        
c        Nodes:
c        
      data tsm6/
     1  0.4719484801783081D-01,0.1560535621012725D+00/
c        
c        Weights:
c        
      data wsm6/
     1  0.9640502775535204D-01,0.1297126527794596D+00/
c        
c        
c  Data for   2 nodes
c        
c        Nodes:
c     
      data tsm5/
     1  0.9407776980350731D-01,0.3109630863764244D+00/
c        
c        Weights:
c        
      data wsm5/
     1  0.1921844979403127D+00,0.2578600108488878D+00/
c        
c        
c  Data for   2 nodes
c        
c        Nodes:
c        
      data tsm4/
     1  0.1887867667318993D+00,0.6289879555150203D+00/
c        
c        Weights:
c        
      data wsm4/
     1  0.3864509579799642D+00,0.5268431163747862D+00/
c        
c        
c  Data for   3 nodes
c        
c        Nodes:
c        
      data tsm3/
     1  0.3001250769898628D+00,0.9417080383819267D+00,
     2  0.1743580070094443D+01/
c        
c        Weights:
c        
      data wsm3/
     1  0.6065146383223869D+00,0.6954038820608324D+00,
     2  0.9492129078768166D+00/
c        
c        
c  Data for   4 nodes
c        
c        Nodes:
c        
      data tsm2/
     1  0.2990364365063186D+00,0.1177848967885806D+01,
     2  0.2312191399448920D+01,0.3706305122526607D+01/
c        
c        Weights:
c        
      data wsm2/
     1  0.6821701038304113D+00,0.1026962494042029D+01,
     2  0.1248563784794822D+01,0.1579847491639718D+01/
c        
c        
c  Data for   6 nodes
c        
c        Nodes:
c        
      data tsm1/
     1  0.7126349633334773D+00,0.2159583827557214D+01,
     2  0.3675994512321710D+01,0.5316610586654979D+01,
     3  0.7134852087854434D+01,0.9188463199735551D+01/
c        
c        Weights:
c        
      data wsm1/
     1  0.1428803875393290D+01,0.1472999784441850D+01,
     2  0.1569272527736830D+01,0.1721314880978768D+01,
     3  0.1927784808862353D+01,0.2233090268448726D+01/
c        
c        
c  Data for   8 nodes
c        
c        Nodes:
c        
      data ts0/
     1  0.6052407376796453D+00,0.2368477654908234D+01,
     2  0.4466034372692647D+01,0.6679740891622117D+01,
     3  0.8989810173214352D+01,0.1139953977041132D+02,
     4  0.1391018528940891D+02,0.1650627617363488D+02/
c        
c        Weights:
c        
      data ws0/
     1  0.1385535326965013D+01,0.2007492808947571D+01,
     2  0.2164012525498241D+01,0.2262675955165567D+01,
     3  0.2361471615918202D+01,0.2468564431856068D+01,
     4  0.2590568777173492D+01,0.2852576364504360D+01/
c        

c        

      if (ilev.ge.nlevels) goto 2000
      
      npw2=npw/2
      
      if (ilev.eq.0) then
         do i=1,npw2
            ts(i+npw2)=ts0(i)
            ws(i+npw2)=ws0(i)
         enddo
      elseif (ilev.eq.-1) then
         do i=1,npw2
            ts(i+npw2)=tsm1(i)
            ws(i+npw2)=wsm1(i)
         enddo
      elseif (ilev.eq.-2) then
         do i=1,npw2
            ts(i+npw2)=tsm2(i)
            ws(i+npw2)=wsm2(i)
         enddo
      elseif (ilev.eq.-3) then
         do i=1,npw2
            ts(i+npw2)=tsm3(i)
            ws(i+npw2)=wsm3(i)
         enddo
      elseif (ilev.eq.-4) then
         do i=1,npw2
            ts(i+npw2)=tsm4(i)
            ws(i+npw2)=wsm4(i)
         enddo
      elseif (ilev.eq.-5) then
         do i=1,npw2
            ts(i+npw2)=tsm5(i)
            ws(i+npw2)=wsm5(i)
         enddo
      elseif (ilev.eq.-6) then
         do i=1,npw2
            ts(i+npw2)=tsm6(i)
            ws(i+npw2)=wsm6(i)
         enddo
      elseif (ilev.eq.-7) then
         do i=1,npw2
            ts(i+npw2)=tsm7(i)
            ws(i+npw2)=wsm7(i)
         enddo
      elseif (ilev.eq.-8) then
         do i=1,npw2
            ts(i+npw2)=tsm8(i)
            ws(i+npw2)=wsm8(i)
         enddo
      elseif (ilev.eq.-9) then
         do i=1,npw2
            ts(i+npw2)=tsm9(i)
            ws(i+npw2)=wsm9(i)
         enddo
      elseif (ilev.eq.-10) then
         do i=1,npw2
            ts(i+npw2)=tsm10(i)
            ws(i+npw2)=wsm10(i)
         enddo
      elseif (ilev.gt.0) then
         do i=1,npw2
            ts(i+npw2)=ts0(i)*2**ilev
            ws(i+npw2)=ws0(i)*2**ilev
         enddo
      endif
      
      do i=1,npw2
         ts(npw2-i+1)=-ts(i+npw2)
         ws(npw2-i+1)=ws(i+npw2)
      enddo
c
      sqrtpi=sqrt(4*atan(1.0d0))
      do i=1,npw
         ws(i)=ws(i)/2/sqrtpi
      enddo

 2000 continue
         
      return
      end 
c
c
c
c
      subroutine get_pwnodes_md3(nlevels,ilev,npw,ws,ts)
C
C     Get planewave expansion weights,nodes for many deltas at the same level ilev
c     3 digits of accuracy
c     data npw0/3,3,3,3,3,3,3,4,5,8,11/
      
      implicit real *8 (a-h,o-z)
      real *8 ws(*),ts(*)
      real *8 wsm10(3),tsm10(3)
      real *8 wsm9(3),tsm9(3)
      real *8 wsm8(3),tsm8(3)
      real *8 wsm7(3),tsm7(3)
      real *8 wsm6(3),tsm6(3)
      real *8 wsm5(3),tsm5(3)
      real *8 wsm4(3),tsm4(3)
      real *8 wsm3(4),tsm3(4)
      real *8 wsm2(5),tsm2(5)
      real *8 wsm1(8),tsm1(8)
      real *8 ws0(11),ts0(11)
c        
c  Data for   3 nodes
c        
c        Nodes:
c        
      data tsm10/
     1  0.3061966198278004D-02,0.9585107530386824D-02,
     2  0.1769868172589105D-01/
c        
c        Weights:
c        
      data wsm10/
     1  0.6185044128706098D-02,0.7039679082612121D-02,
     2  0.9674297371467726D-02/
c        
c        
c  Data for   3 nodes
c        
c        Nodes:
c        
      data tsm9/
     1  0.3228978122706957D-02,0.1428227512925793D-01,
     2  0.3001723189800723D-01/
c        
c        Weights:
c        
      data wsm9/
     1  0.7918213993540806D-02,0.1333099554900802D-01,
     2  0.1908058731923880D-01/
c        
c        
c  Data for   3 nodes
c        
c        Nodes:
c        
      data tsm8/
     1  0.1199665039018494D-01,0.3746892400739685D-01,
     2  0.6899651715920062D-01/
c        
c        Weights:
c        
      data wsm8/
     1  0.2421869817473647D-01,0.2740035192915402D-01,
     2  0.3763204405980771D-01/
c        
c        
c  Data for   3 nodes
c        
c        Nodes:
c        
      data tsm7/
     1  0.2397354422399662D-01,0.7487175854857331D-01,
     2  0.1378509835483378D+00/
c        
c        Weights:
c        
      data wsm7/
     1  0.4839688080012997D-01,0.5474516397071366D-01,
     2  0.7515799806811344D-01/
c        
c        
c  Data for   3 nodes
c        
c        Nodes:
c        
      data tsm6/
     1  0.4902032501332574D-01,0.1535307775311054D+00,
     2  0.2837465098912746D+00/
c        
c        Weights:
c        
      data wsm6/
     1  0.9903094552808132D-01,0.1128793439618068D+00,
     2  0.1553309751739821D+00/
c        
c        
c  Data for   3 nodes
c        
c        Nodes:
c        
      data tsm5/
     1  0.9805804228918055D-01,0.3075514067599542D+00,
     2  0.5698872379665393D+00/
c        
c        Weights:
c        
      data wsm5/
     1  0.1981618985714773D+00,0.2267909372249381D+00,
     2  0.3135440601051060D+00/
c        
c        
c  Data for   3 nodes
c        
c        Nodes:
c        
      data tsm4/
     1  0.1892578332215519D+00,0.5929467207083788D+00,
     2  0.1095283991303266D+01/
c        
c        Weights:
c        
      data wsm4/
     1  0.3823580730609019D+00,0.4362577644791776D+00,
     2  0.5966633337738166D+00/
c        
c        
c  Data for   4 nodes
c        
c        Nodes:
c        
      data tsm3/
     1  0.2013147683812810D+00,0.7697833529445938D+00,
     2  0.1506524854596830D+01,0.2447608045285776D+01/
c        
c        Weights:
c        
      data wsm3/
     1  0.4505609199204278D+00,0.6622069595817983D+00,
     2  0.8193972467829352D+00,0.1100657763741081D+01/
c        
c        
c  Data for   5 nodes
c        
c        Nodes:
c        
      data tsm2/
     1  0.5476656572169173D+00,0.1668371907085270D+01,
     2  0.2871255179431555D+01,0.4227323321327876D+01,
     3  0.5845545710754728D+01/
c        
c        Weights:
c        
      data wsm2/
     1  0.1099444296033681D+01,0.1151316040239638D+01,
     2  0.1266180604555044D+01,0.1462379465201503D+01,
     3  0.1825111180615093D+01/
c        
c        
c  Data for   8 nodes
c        
c        Nodes:
c        
      data tsm1/
     1  0.3606324021765846D+00,0.1711744658446535D+01,
     2  0.3319101670277704D+01,0.5015089250787778D+01,
     3  0.6812155417438572D+01,0.8737685943298711D+01,
     4  0.1082674832379779D+02,0.1313920559967450D+02/
c        
c        Weights:
c        
      data wsm1/
     1  0.9489427999932037D+00,0.1550655632750167D+01,
     2  0.1652659707949705D+01,0.1742406522146077D+01,
     3  0.1856416261097314D+01,0.2000556462269979D+01,
     4  0.2189806576785785D+01,0.2497096974147865D+01/
c        
c        
c  Data for  11 nodes
c        
c        Nodes:
c        
      data ts0/
     1  0.1095377436338480D+01,0.3292942978694948D+01,
     2  0.5510991398644411D+01,0.7762964648916271D+01,
     3  0.1006113638037723D+02,0.1241551793703057D+02,
     4  0.1483353708394748D+02,0.1732097439308589D+02,
     5  0.1988323932298671D+02,0.2252668776597777D+02,
     6  0.2525927074999686D+02/
c        
c        Weights:
c        
      data ws0/
     1  0.2191890488645997D+01,0.2205530503606007D+01,
     2  0.2232843326031164D+01,0.2273220872505014D+01,
     3  0.2324933044852573D+01,0.2385376177434400D+01,
     4  0.2452443303010661D+01,0.2525673045653326D+01,
     5  0.2607764998200619D+01,0.2713738182920184D+01,
     6  0.2999278612283102D+01/
c        

c        

      if (ilev.ge.nlevels) goto 2000
      
      npw2=npw/2
      
      if (ilev.eq.0) then
         do i=1,npw2
            ts(i+npw2)=ts0(i)
            ws(i+npw2)=ws0(i)
         enddo
      elseif (ilev.eq.-1) then
         do i=1,npw2
            ts(i+npw2)=tsm1(i)
            ws(i+npw2)=wsm1(i)
         enddo
      elseif (ilev.eq.-2) then
         do i=1,npw2
            ts(i+npw2)=tsm2(i)
            ws(i+npw2)=wsm2(i)
         enddo
      elseif (ilev.eq.-3) then
         do i=1,npw2
            ts(i+npw2)=tsm3(i)
            ws(i+npw2)=wsm3(i)
         enddo
      elseif (ilev.eq.-4) then
         do i=1,npw2
            ts(i+npw2)=tsm4(i)
            ws(i+npw2)=wsm4(i)
         enddo
      elseif (ilev.eq.-5) then
         do i=1,npw2
            ts(i+npw2)=tsm5(i)
            ws(i+npw2)=wsm5(i)
         enddo
      elseif (ilev.eq.-6) then
         do i=1,npw2
            ts(i+npw2)=tsm6(i)
            ws(i+npw2)=wsm6(i)
         enddo
      elseif (ilev.eq.-7) then
         do i=1,npw2
            ts(i+npw2)=tsm7(i)
            ws(i+npw2)=wsm7(i)
         enddo
      elseif (ilev.eq.-8) then
         do i=1,npw2
            ts(i+npw2)=tsm8(i)
            ws(i+npw2)=wsm8(i)
         enddo
      elseif (ilev.eq.-9) then
         do i=1,npw2
            ts(i+npw2)=tsm9(i)
            ws(i+npw2)=wsm9(i)
         enddo
      elseif (ilev.eq.-10) then
         do i=1,npw2
            ts(i+npw2)=tsm10(i)
            ws(i+npw2)=wsm10(i)
         enddo
      elseif (ilev.gt.0) then
         do i=1,npw2
            ts(i+npw2)=ts0(i)*2**ilev
            ws(i+npw2)=ws0(i)*2**ilev
         enddo
      endif
      
      do i=1,npw2
         ts(npw2-i+1)=-ts(i+npw2)
         ws(npw2-i+1)=ws(i+npw2)
      enddo
c
      sqrtpi=sqrt(4*atan(1.0d0))
      do i=1,npw
         ws(i)=ws(i)/2/sqrtpi
      enddo

 2000 continue
         
      return
      end 
c
c
c
c
      subroutine get_pwnodes_md4(nlevels,ilev,npw,ws,ts)
C
C     Get planewave expansion weights,nodes for many deltas at the same level ilev
c     4 digits of accuracy
c     data npw0/4,4,4,4,3,4,4,5,7,10,15/

      
      implicit real *8 (a-h,o-z)
      real *8 ws(*),ts(*)
      real *8 wsm10(4),tsm10(4)
      real *8 wsm9(4),tsm9(4)
      real *8 wsm8(4),tsm8(4)
      real *8 wsm7(4),tsm7(4)
      real *8 wsm6(3),tsm6(3)
      real *8 wsm5(4),tsm5(4)
      real *8 wsm4(4),tsm4(4)
      real *8 wsm3(5),tsm3(5)
      real *8 wsm2(7),tsm2(7)
      real *8 wsm1(10),tsm1(10)
      real *8 ws0(15),ts0(15)
c        
c  Data for   4 nodes
c        
c        Nodes:
c        
      data tsm10/
     1  0.2454579259354462D-02,0.7914725056463632D-02,
     2  0.1453599570623560D-01,0.2318809599861219D-01/
c        
c        Weights:
c        
      data wsm10/
     1  0.5022078113841440D-02,0.5973186483531776D-02,
     2  0.7387341054821006D-02,0.1042137929779285D-01/
c        
c        
c  Data for   4 nodes
c        
c        Nodes:
c        
      data tsm9/
     1  0.3956979483433365D-02,0.1457814442131598D-01,
     2  0.2815596719708853D-01,0.4570523836153013D-01/
c        
c        Weights:
c        
      data wsm9/
     1  0.8666721721464656D-02,0.1224768491726788D-01,
     2  0.1507375759294738D-01,0.2104575235543538D-01/
c        
c        
c  Data for   4 nodes
c        
c        Nodes:
c        
      data tsm8/
     1  0.7924614723055907D-02,0.2916044679236733D-01,
     2  0.5630155416768811D-01,0.9138113218767052D-01/
c        
c        Weights:
c        
      data wsm8/
     1  0.1734450558770798D-01,0.2448138504865639D-01,
     2  0.3013195800982963D-01,0.4206727639984143D-01/
c        
c        
c  Data for   4 nodes
c        
c        Nodes:
c        
      data tsm7/
     1  0.1643997036576240D-01,0.5890263614365335D-01,
     2  0.1129074372078578D+00,0.1828528224164818D+00/
c        
c        Weights:
c        
      data wsm7/
     1  0.3543288988152905D-01,0.4864402483344796D-01,
     2  0.6003449250319001D-01,0.8391254917368118D-01/
c        
c        
c  Data for   3 nodes
c        
c        Nodes:
c        
      data tsm6/
     1  0.5532210478330254D-01,0.1728208712392092D+00,
     2  0.3182766983557295D+00/
c        
c        Weights:
c        
      data wsm6/
     1  0.1116892116339322D+00,0.1264281599733733D+00,
     2  0.1734917070055272D+00/
c        
c        
c  Data for   4 nodes
c        
c        Nodes:
c        
      data tsm5/
     1  0.9648326770767621D-01,0.2961830737437161D+00,
     2  0.5199395954046986D+00,0.8021643401623078D+00/
c        
c        Weights:
c        
      data wsm5/
     1  0.1940317186420348D+00,0.2080882740422495D+00,
     2  0.2445066312558223D+00,0.3357661114884733D+00/
c        
c        
c  Data for   4 nodes
c        
c        Nodes:
c        
      data tsm4/
     1  0.1913753079294980D+00,0.5882069715530125D+00,
     2  0.1035019453634884D+01,0.1599482842944753D+01/
c        
c        Weights:
c        
      data wsm4/
     1  0.3849776734602839D+00,0.4143631963335372D+00,
     2  0.4893866419315878D+00,0.6692731335998000D+00/
c        
c        
c  Data for   5 nodes
c        
c        Nodes:
c        
      data tsm3/
     1  0.3201289584719097D+00,0.9761045020878404D+00,
     2  0.1685178346801574D+01,0.2503098836005693D+01,
     3  0.3528668008350171D+01/
c        
c        Weights:
c        
      data wsm3/
     1  0.6427817204001319D+00,0.6751896173583283D+00,
     2  0.7520240057176875D+00,0.8986079958467517D+00,
     3  0.1195944442729404D+01/
c        
c        
c  Data for   7 nodes
c        
c        Nodes:
c        
      data tsm2/
     1  0.1917216588995281D+00,0.1155141599378981D+01,
     2  0.2288745025300090D+01,0.3500402466920777D+01,
     3  0.4825355395682612D+01,0.6318240762349509D+01,
     4  0.8081955825172132D+01/
c        
c        Weights:
c        
      data wsm2/
     1  0.6137205548081583D+00,0.1096825727232116D+01,
     2  0.1168789201096179D+01,0.1260788513389286D+01,
     3  0.1397911525062111D+01,0.1602949907821591D+01,
     4  0.1976684224780406D+01/
c        
c        
c  Data for  10 nodes
c        
c        Nodes:
c        
      data tsm1/
     1  0.2678767855786053D+00,0.1703315977813667D+01,
     2  0.3356442105768776D+01,0.5062232350066859D+01,
     3  0.6830288821092096D+01,0.8679238552759932D+01,
     4  0.1062918047249195D+02,0.1270280006504918D+02,
     5  0.1493317622356338D+02,0.1738379141291341D+02/
c        
c        Weights:
c        
      data wsm1/
     1  0.9008621085365320D+00,0.1619793017432331D+01,
     2  0.1679685688499582D+01,0.1734073888921819D+01,
     3  0.1805223665726511D+01,0.1896020593941126D+01,
     4  0.2007573024402137D+01,0.2145139771781141D+01,
     5  0.2327795328737079D+01,0.2637146560197285D+01/
c        
c        
c  Data for  15 nodes
c        
c        Nodes:
c        
      data ts0/
     1  0.1099391695519098D+01,0.3301938956704258D+01,
     2  0.5515800617579211D+01,0.7748550691621873D+01,
     3  0.1000766674549858D+02,0.1230026354841533D+02,
     4  0.1463275760008210D+02,0.1701065371417228D+02,
     5  0.1943863729643465D+02,0.2192104056065905D+02,
     6  0.2446267695241818D+02,0.2707015958656086D+02,
     7  0.2975416988803943D+02,0.3253471674998087D+02,
     8  0.3552536747302340D+02/
c        
c        Weights:
c        
      data ws0/
     1  0.2199410257220401D+01,0.2206942607867374D+01,
     2  0.2222045248884205D+01,0.2244706439678685D+01,
     3  0.2274719181277254D+01,0.2311551379961361D+01,
     4  0.2354360306380531D+01,0.2402221307016964D+01,
     5  0.2454492036433119D+01,0.2511190469292878D+01,
     6  0.2573412496812777D+01,0.2644092032770288D+01,
     7  0.2730068662561151D+01,0.2852104070551538D+01,
     8  0.3491399524862274D+01/
c        

c        

      if (ilev.ge.nlevels) goto 2000
      
      npw2=npw/2
      
      if (ilev.eq.0) then
         do i=1,npw2
            ts(i+npw2)=ts0(i)
            ws(i+npw2)=ws0(i)
         enddo
      elseif (ilev.eq.-1) then
         do i=1,npw2
            ts(i+npw2)=tsm1(i)
            ws(i+npw2)=wsm1(i)
         enddo
      elseif (ilev.eq.-2) then
         do i=1,npw2
            ts(i+npw2)=tsm2(i)
            ws(i+npw2)=wsm2(i)
         enddo
      elseif (ilev.eq.-3) then
         do i=1,npw2
            ts(i+npw2)=tsm3(i)
            ws(i+npw2)=wsm3(i)
         enddo
      elseif (ilev.eq.-4) then
         do i=1,npw2
            ts(i+npw2)=tsm4(i)
            ws(i+npw2)=wsm4(i)
         enddo
      elseif (ilev.eq.-5) then
         do i=1,npw2
            ts(i+npw2)=tsm5(i)
            ws(i+npw2)=wsm5(i)
         enddo
      elseif (ilev.eq.-6) then
         do i=1,npw2
            ts(i+npw2)=tsm6(i)
            ws(i+npw2)=wsm6(i)
         enddo
      elseif (ilev.eq.-7) then
         do i=1,npw2
            ts(i+npw2)=tsm7(i)
            ws(i+npw2)=wsm7(i)
         enddo
      elseif (ilev.eq.-8) then
         do i=1,npw2
            ts(i+npw2)=tsm8(i)
            ws(i+npw2)=wsm8(i)
         enddo
      elseif (ilev.eq.-9) then
         do i=1,npw2
            ts(i+npw2)=tsm9(i)
            ws(i+npw2)=wsm9(i)
         enddo
      elseif (ilev.eq.-10) then
         do i=1,npw2
            ts(i+npw2)=tsm10(i)
            ws(i+npw2)=wsm10(i)
         enddo
      elseif (ilev.gt.0) then
         do i=1,npw2
            ts(i+npw2)=ts0(i)*2**ilev
            ws(i+npw2)=ws0(i)*2**ilev
         enddo
      endif
      
      do i=1,npw2
         ts(npw2-i+1)=-ts(i+npw2)
         ws(npw2-i+1)=ws(i+npw2)
      enddo
c
      sqrtpi=sqrt(4*atan(1.0d0))
      do i=1,npw
         ws(i)=ws(i)/2/sqrtpi
      enddo

 2000 continue
         
      return
      end 
c
c
c
c
      subroutine get_pwnodes_md5(nlevels,ilev,npw,ws,ts)
C
C     Get planewave expansion weights,nodes for many deltas at the same level ilev
c     5 digits of accuracy
c     data npw0/4,4,4,4,4,4,5,6,8,12,19/

      
      implicit real *8 (a-h,o-z)
      real *8 ws(*),ts(*)
      real *8 wsm10(4),tsm10(4)
      real *8 wsm9(4),tsm9(4)
      real *8 wsm8(4),tsm8(4)
      real *8 wsm7(4),tsm7(4)
      real *8 wsm6(4),tsm6(4)
      real *8 wsm5(4),tsm5(4)
      real *8 wsm4(5),tsm4(5)
      real *8 wsm3(6),tsm3(6)
      real *8 wsm2(8),tsm2(8)
      real *8 wsm1(12),tsm1(12)
      real *8 ws0(19),ts0(19)
c        
c  Data for   4 nodes
c        
c        Nodes:
c        
      data tsm10/
     1  0.3380988427457265D-02,0.1036886531055555D-01,
     2  0.1816510936355607D-01,0.2795839841396395D-01/
c        
c        Weights:
c        
      data wsm10/
     1  0.6797740089643270D-02,0.7269184771949468D-02,
     2  0.8497286372178638D-02,0.1165164524505916D-01/
c        
c        
c  Data for   4 nodes
c        
c        Nodes:
c        
      data tsm9/
     1  0.6761314267092079D-02,0.2073561390755237D-01,
     2  0.3632616892174672D-01,0.5590966120251310D-01/
c        
c        Weights:
c        
      data wsm9/
     1  0.1359413496579986D-01,0.1453675159620994D-01,
     2  0.1699222022373144D-01,0.2329920966454406D-01/
c        
c        
c  Data for   4 nodes
c        
c        Nodes:
c        
      data tsm8/
     1  0.1351783517412260D-01,0.4145597914705831D-01,
     2  0.7262338310705387D-01,0.1117685780792298D+00/
c        
c        Weights:
c        
      data wsm8/
     1  0.2717854836624041D-01,0.2906194258589471D-01,
     2  0.3396785452193657D-01,0.4656939851815391D-01/
c        
c        
c  Data for   4 nodes
c        
c        Nodes:
c        
      data tsm7/
     1  0.2701613159383656D-01,0.8285235612177527D-01,
     2  0.1451423525542442D+00,0.2233652752733256D+00/
c        
c        Weights:
c        
      data wsm7/
     1  0.5431787200937321D-01,0.5808248603529707D-01,
     2  0.6788529676767031D-01,0.9304086327224975D-01/
c        
c        
c  Data for   4 nodes
c        
c        Nodes:
c        
      data tsm6/
     1  0.5413711940487693D-01,0.1661043888628531D+00,
     2  0.2912894944478335D+00,0.4489573495001835D+00/
c        
c        Weights:
c        
      data wsm6/
     1  0.1088586225784690D+00,0.1165680890936344D+00,
     2  0.1366333553948986D+00,0.1876918832532168D+00/
c        
c        
c  Data for   4 nodes
c        
c        Nodes:
c        
      data tsm5/
     1  0.1076411248637149D+00,0.3304652411813703D+00,
     2  0.5802237054145779D+00,0.8952270748015010D+00/
c        
c        Weights:
c        
      data wsm5/
     1  0.2164752410476708D+00,0.2322202342271128D+00,
     2  0.2729644293479207D+00,0.3745658184022514D+00/
c        
c        
c  Data for   5 nodes
c        
c        Nodes:
c        
      data tsm4/
     1  0.1838053621865438D+00,0.5602663574553877D+00,
     2  0.9666627457242540D+00,0.1436009961990249D+01,
     3  0.2034250505964664D+01/
c        
c        Weights:
c        
      data wsm4/
     1  0.3690334755918898D+00,0.3872697450027133D+00,
     2  0.4307905013558621D+00,0.5176332337811219D+00,
     3  0.7094934194903331D+00/
c        
c        
c  Data for   6 nodes
c        
c        Nodes:
c        
      data tsm3/
     1  0.1948931312591489D+00,0.7804889055060583D+00,
     2  0.1495594140101966D+01,0.2293297720281673D+01,
     3  0.3213025827435240D+01,0.4351348609355308D+01/
c        
c        Weights:
c        
      data wsm3/
     1  0.4519908034881868D+00,0.6733581404827906D+00,
     2  0.7533718513495882D+00,0.8488422404622756D+00,
     3  0.1004602763059866D+01,0.1320572749720515D+01/
c        
c        
c  Data for   8 nodes
c        
c        Nodes:
c        
      data tsm2/
     1  0.5487635555165695D+00,0.1655863438547989D+01,
     2  0.2792955698049153D+01,0.3984194379717679D+01,
     3  0.5259382374163989D+01,0.6655560141533735D+01,
     4  0.8224242372729751D+01,0.1007110891160430D+02/
c        
c        Weights:
c        
      data wsm2/
     1  0.1099101328499103D+01,0.1118459537874515D+01,
     2  0.1159717205664913D+01,0.1227711781594306D+01,
     3  0.1328762778712679D+01,0.1471730058850644D+01,
     4  0.1680950152668015D+01,0.2068262444622565D+01/
c        
c        
c  Data for  12 nodes
c        
c        Nodes:
c        
      data tsm1/
     1  0.8261297862753223D+00,0.2484062126521364D+01,
     2  0.4159210351299820D+01,0.5863647885030486D+01,
     3  0.7610117910075448D+01,0.9411892395296670D+01,
     4  0.1128248350521515D+02,0.1323572283260376D+02,
     5  0.1528703280668070D+02,0.1745701162283403D+02,
     6  0.1978003832243837D+02,0.2232717102690759D+02/
c        
c        Weights:
c        
      data wsm1/
     1  0.1653201600596402D+01,0.1664581069521072D+01,
     2  0.1687725021685002D+01,0.1723276532235119D+01,
     3  0.1771881735513722D+01,0.1833920404278539D+01,
     4  0.1909549509429474D+01,0.1999449427376508D+01,
     5  0.2106501342582606D+01,0.2239015209945584D+01,
     6  0.2419615582055371D+01,0.2736276076197787D+01/
c        
c        
c  Data for  19 nodes
c        
c        Nodes:
c        
      data ts0/
     1  0.1070474156635697D+01,0.3213737628498020D+01,
     2  0.5363968668974327D+01,0.7525879559463320D+01,
     3  0.9704259531656666D+01,0.1190395404222104D+02,
     4  0.1412979021813044D+02,0.1638644527497964D+02,
     5  0.1867828290743335D+02,0.2100921491294123D+02,
     6  0.2338265800240509D+02,0.2580163527054255D+02,
     7  0.2826903335252237D+02,0.3078800350378782D+02,
     8  0.3336251070860755D+02,0.3599808850887506D+02,
     9  0.3870291630348485D+02,0.4148899763928926D+02,
     *  0.4436796875969318D+02/
c        
c        Weights:
c        
      data ws0/
     1  0.2141333795100472D+01,0.2145967900148252D+01,
     2  0.2155279258682087D+01,0.2169341082684115D+01,
     3  0.2188227778857844D+01,0.2211968245249656D+01,
     4  0.2240486967961096D+01,0.2273553790648739D+01,
     5  0.2310775995682530D+01,0.2351659264450075D+01,
     6  0.2395733861169448D+01,0.2442712080530474D+01,
     7  0.2492642271401012D+01,0.2546060354886698D+01,
     8  0.2604201712706816D+01,0.2669472326302595D+01,
     9  0.2746989026199317D+01,0.2852504328206094D+01,
     *  0.3118574230653961D+01/
c        

c        

      if (ilev.ge.nlevels) goto 2000
      
      npw2=npw/2
      
      if (ilev.eq.0) then
         do i=1,npw2
            ts(i+npw2)=ts0(i)
            ws(i+npw2)=ws0(i)
         enddo
      elseif (ilev.eq.-1) then
         do i=1,npw2
            ts(i+npw2)=tsm1(i)
            ws(i+npw2)=wsm1(i)
         enddo
      elseif (ilev.eq.-2) then
         do i=1,npw2
            ts(i+npw2)=tsm2(i)
            ws(i+npw2)=wsm2(i)
         enddo
      elseif (ilev.eq.-3) then
         do i=1,npw2
            ts(i+npw2)=tsm3(i)
            ws(i+npw2)=wsm3(i)
         enddo
      elseif (ilev.eq.-4) then
         do i=1,npw2
            ts(i+npw2)=tsm4(i)
            ws(i+npw2)=wsm4(i)
         enddo
      elseif (ilev.eq.-5) then
         do i=1,npw2
            ts(i+npw2)=tsm5(i)
            ws(i+npw2)=wsm5(i)
         enddo
      elseif (ilev.eq.-6) then
         do i=1,npw2
            ts(i+npw2)=tsm6(i)
            ws(i+npw2)=wsm6(i)
         enddo
      elseif (ilev.eq.-7) then
         do i=1,npw2
            ts(i+npw2)=tsm7(i)
            ws(i+npw2)=wsm7(i)
         enddo
      elseif (ilev.eq.-8) then
         do i=1,npw2
            ts(i+npw2)=tsm8(i)
            ws(i+npw2)=wsm8(i)
         enddo
      elseif (ilev.eq.-9) then
         do i=1,npw2
            ts(i+npw2)=tsm9(i)
            ws(i+npw2)=wsm9(i)
         enddo
      elseif (ilev.eq.-10) then
         do i=1,npw2
            ts(i+npw2)=tsm10(i)
            ws(i+npw2)=wsm10(i)
         enddo
      elseif (ilev.gt.0) then
         do i=1,npw2
            ts(i+npw2)=ts0(i)*2**ilev
            ws(i+npw2)=ws0(i)*2**ilev
         enddo
      endif
      
      do i=1,npw2
         ts(npw2-i+1)=-ts(i+npw2)
         ws(npw2-i+1)=ws(i+npw2)
      enddo
c
      sqrtpi=sqrt(4*atan(1.0d0))
      do i=1,npw
         ws(i)=ws(i)/2/sqrtpi
      enddo

 2000 continue
         
      return
      end 
c
c
c
c
      subroutine get_pwnodes_md6(nlevels,ilev,npw,ws,ts)
C
C     Get planewave expansion weights,nodes for many deltas at the same level ilev
c     6 digits of accuracy
c      
      implicit real *8 (a-h,o-z)
      real *8 ws(*),ts(*)
      real *8 wsm10(5),tsm10(5)
      real *8 wsm9(5),tsm9(5)
      real *8 wsm8(5),tsm8(5)
      real *8 wsm7(5),tsm7(5)
      real *8 wsm6(5),tsm6(5)
      real *8 wsm5(5),tsm5(5)
      real *8 wsm4(6),tsm4(6)
      real *8 wsm3(7),tsm3(7)
      real *8 wsm2(9),tsm2(9)
      real *8 wsm1(14),tsm1(14)
      real *8 ws0(22),ts0(22)
c        
c  Data for   5 nodes
c        
c        Nodes:
c        
      data tsm10/
     1  0.3186743201261397D-02,0.9705391828218125D-02,
     2  0.1671102365646551D-01,0.2474676217357151D-01,
     3  0.3499603913532293D-01/
c        
c        Weights:
c        
      data wsm10/
     1  0.6396912362106923D-02,0.6695172480266405D-02,
     2  0.7401450568252634D-02,0.8841959662652672D-02,
     3  0.1223668635211297D-01/
c        
c        
c  Data for   5 nodes
c        
c        Nodes:
c        
      data tsm9/
     1  0.6621620157662806D-02,0.2014132301104222D-01,
     2  0.3459177686355034D-01,0.5101979497398396D-01,
     3  0.7173352293274046D-01/
c        
c        Weights:
c        
      data wsm9/
     1  0.1328779012851835D-01,0.1385644189554157D-01,
     2  0.1520884757320645D-01,0.1798110023478758D-01,
     3  0.2459373095269948D-01/
c        
c        
c  Data for   5 nodes
c        
c        Nodes:
c        
      data tsm8/
     1  0.1324284486741650D-01,0.4028137241593716D-01,
     2  0.6918079416217097D-01,0.1020327067699522D+00,
     3  0.1434472944722616D+00/
c        
c        Weights:
c        
      data wsm8/
     1  0.2657477945203070D-01,0.2771185878847594D-01,
     2  0.3041544946010178D-01,0.3595527390200643D-01,
     3  0.4916593783755301D-01/
c        
c        
c  Data for   5 nodes
c        
c        Nodes:
c        
      data tsm7/
     1  0.2662613103380669D-01,0.8100283534698077D-01,
     2  0.1391574217585120D+00,0.2052972873680076D+00,
     3  0.2886114692557206D+00/
c        
c        Weights:
c        
      data wsm7/
     1  0.5343354538093600D-01,0.5574540016791833D-01,
     2  0.6122551595718347D-01,0.7238367933360329D-01,
     3  0.9880657627677991D-01/
c        
c        
c  Data for   5 nodes
c        
c        Nodes:
c        
      data tsm6/
     1  0.5286609117245560D-01,0.1608529954476055D+00,
     2  0.2764439998493440D+00,0.4082111893143786D+00,
     3  0.5748909407632388D+00/
c        
c        Weights:
c        
      data wsm6/
     1  0.1060952824754609D+00,0.1107352957438620D+00,
     2  0.1217945409800223D+00,0.1444588053438139D+00,
     3  0.1981223061738356D+00/
c        
c        
c  Data for   5 nodes
c        
c        Nodes:
c        
      data tsm5/
     1  0.1063156088162535D+00,0.3236990645592296D+00,
     2  0.5570549184669225D+00,0.8239312625303060D+00,
     3  0.1161177501311156D+01/
c        
c        Weights:
c        
      data wsm5/
     1  0.2133968748371036D+00,0.2231715170396798D+00,
     2  0.2463175853251708D+00,0.2928025828713196D+00,
     3  0.3995530847637025D+00/
c        
c        
c  Data for   6 nodes
c        
c        Nodes:
c        
      data tsm4/
     1  0.9074379970263043D-01,0.4272308935138848D+00,
     2  0.8360645779078744D+00,0.1288337775494350D+01,
     3  0.1811225081153655D+01,0.2470236858057746D+01/
c        
c        Weights:
c        
      data wsm4/
     1  0.2367240233915840D+00,0.3886899434738869D+00,
     2  0.4280845746339439D+00,0.4809904405637751D+00,
     3  0.5742733336708470D+00,0.7765099283502653D+00/
c        
c        
c  Data for   7 nodes
c        
c        Nodes:
c        
      data tsm3/
     1  0.3515493357835718D+00,0.1063210484339505D+01,
     2  0.1802001432369091D+01,0.2590910378595406D+01,
     3  0.3460844999648564D+01,0.4458440457927277D+01,
     4  0.5684154231062625D+01/
c        
c        Weights:
c        
      data wsm3/
     1  0.7045019914482193D+00,0.7218654955531798D+00,
     2  0.7594941285229715D+00,0.8233806089300912D+00,
     3  0.9238614667962268D+00,0.1085529544392842D+01,
     4  0.1419180206692230D+01/
c        
c        
c  Data for   9 nodes
c        
c        Nodes:
c        
      data tsm2/
     1  0.5801084192587924D+00,0.1748414437020543D+01,
     2  0.2941706524900200D+01,0.4178949419616425D+01,
     3  0.5482230427903951D+01,0.6877645364098862D+01,
     4  0.8397856400478949D+01,0.1009265317382069D+02,
     5  0.1207356359164341D+02/
c        
c        Weights:
c        
      data wsm2/
     1  0.1161552893030750D+01,0.1177852230095766D+01,
     2  0.1211874878714621D+01,0.1266280872109370D+01,
     3  0.1344600548807025D+01,0.1451534954236905D+01,
     4  0.1596666323539673D+01,0.1808962585969095D+01,
     5  0.2212368208806546D+01/
c        
c        
c  Data for  14 nodes
c        
c        Nodes:
c        
      data tsm1/
     1  0.8372503232931269D+00,0.2515972442957355D+01,
     2  0.4207467881199861D+01,0.5920591782806681D+01,
     3  0.7664623319950377D+01,0.9449265209308063D+01,
     4  0.1128451910822071D+02,0.1318054812973855D+02,
     5  0.1514778669615706D+02,0.1719766463861083D+02,
     6  0.1934441668100972D+02,0.2160892088564468D+02,
     7  0.2402761600006155D+02,0.2667904415101633D+02/
c        
c        Weights:
c        
      data wsm1/
     1  0.1675202357136789D+01,0.1683663830061612D+01,
     2  0.1700800763254662D+01,0.1726993540030479D+01,
     3  0.1762688984421021D+01,0.1808264696017835D+01,
     4  0.1863936083466716D+01,0.1929850166495683D+01,
     5  0.2006504222299840D+01,0.2095568356142992D+01,
     6  0.2201302653740221D+01,0.2333543670426394D+01,
     7  0.2517014386012851D+01,0.2851714693443992D+01/
c        
c        
c  Data for  22 nodes
c        
c        Nodes:
c        
      data ts0/
     1  0.1105920556779977D+01,0.3319444786931895D+01,
     2  0.5538024636562354D+01,0.7765048609265154D+01,
     3  0.1000392252416003D+02,0.1225805712021132D+02,
     4  0.1453083813480531D+02,0.1682557878360741D+02,
     5  0.1914546089529526D+02,0.2149347844073872D+02,
     6  0.2387240255563413D+02,0.2628478712018921D+02,
     7  0.2873302826050859D+02,0.3121948336774793D+02,
     8  0.3374665089025162D+02,0.3631741589065327D+02,
     9  0.3893538162501314D+02,0.4160534098173623D+02,
     *  0.4433401343508787D+02,0.4713133308003184D+02,
     1  0.5001295150174944D+02,0.5300964382622754D+02/
c        
c        Weights:
c        
      data ws0/
     1  0.2212121522267698D+01,0.2215488847369463D+01,
     2  0.2222235569600474D+01,0.2232380213140591D+01,
     3  0.2245936579384897D+01,0.2262897726223185D+01,
     4  0.2283217553694604D+01,0.2306795099539102D+01,
     5  0.2333468863849065D+01,0.2363028237293543D+01,
     6  0.2395245077527914D+01,0.2429922274653722D+01,
     7  0.2466951750213254D+01,0.2506375019171602D+01,
     8  0.2548445753974315D+01,0.2593704523042859D+01,
     9  0.2643093427846662D+01,0.2698180989485878D+01,
     *  0.2761718090841037D+01,0.2839483722284254D+01,
     1  0.2950112580651863D+01,0.3264619113492975D+01/
c        

      if (ilev.ge.nlevels) goto 2000
      
      npw2=npw/2
      
      if (ilev.eq.0) then
         do i=1,npw2
            ts(i+npw2)=ts0(i)
            ws(i+npw2)=ws0(i)
         enddo
      elseif (ilev.eq.-1) then
         do i=1,npw2
            ts(i+npw2)=tsm1(i)
            ws(i+npw2)=wsm1(i)
         enddo
      elseif (ilev.eq.-2) then
         do i=1,npw2
            ts(i+npw2)=tsm2(i)
            ws(i+npw2)=wsm2(i)
         enddo
      elseif (ilev.eq.-3) then
         do i=1,npw2
            ts(i+npw2)=tsm3(i)
            ws(i+npw2)=wsm3(i)
         enddo
      elseif (ilev.eq.-4) then
         do i=1,npw2
            ts(i+npw2)=tsm4(i)
            ws(i+npw2)=wsm4(i)
         enddo
      elseif (ilev.eq.-5) then
         do i=1,npw2
            ts(i+npw2)=tsm5(i)
            ws(i+npw2)=wsm5(i)
         enddo
      elseif (ilev.eq.-6) then
         do i=1,npw2
            ts(i+npw2)=tsm6(i)
            ws(i+npw2)=wsm6(i)
         enddo
      elseif (ilev.eq.-7) then
         do i=1,npw2
            ts(i+npw2)=tsm7(i)
            ws(i+npw2)=wsm7(i)
         enddo
      elseif (ilev.eq.-8) then
         do i=1,npw2
            ts(i+npw2)=tsm8(i)
            ws(i+npw2)=wsm8(i)
         enddo
      elseif (ilev.eq.-9) then
         do i=1,npw2
            ts(i+npw2)=tsm9(i)
            ws(i+npw2)=wsm9(i)
         enddo
      elseif (ilev.eq.-10) then
         do i=1,npw2
            ts(i+npw2)=tsm10(i)
            ws(i+npw2)=wsm10(i)
         enddo
      elseif (ilev.gt.0) then
         do i=1,npw2
            ts(i+npw2)=ts0(i)*2**ilev
            ws(i+npw2)=ws0(i)*2**ilev
         enddo
      endif
      
      do i=1,npw2
         ts(npw2-i+1)=-ts(i+npw2)
         ws(npw2-i+1)=ws(i+npw2)
      enddo
c
      sqrtpi=sqrt(4*atan(1.0d0))
      do i=1,npw
         ws(i)=ws(i)/2/sqrtpi
      enddo

 2000 continue
         
      return
      end 
c
c
c
c
      subroutine get_pwnodes_md7(nlevels,ilev,npw,ws,ts)
C
C     Get planewave expansion weights,nodes for many deltas at the same level ilev
c     7 digits of accuracy
c     data npw0/5,6,6,6,5,6,7,8,11,16,26/

      
      implicit real *8 (a-h,o-z)
      real *8 ws(*),ts(*)
      real *8 wsm10(5),tsm10(5)
      real *8 wsm9(6),tsm9(6)
      real *8 wsm8(6),tsm8(6)
      real *8 wsm7(6),tsm7(6)
      real *8 wsm6(5),tsm6(5)
      real *8 wsm5(6),tsm5(6)
      real *8 wsm4(7),tsm4(7)
      real *8 wsm3(8),tsm3(8)
      real *8 wsm2(11),tsm2(11)
      real *8 wsm1(16),tsm1(16)
      real *8 ws0(26),ts0(26)
c        
c  Data for   5 nodes
c        
c        Nodes:
c        
      data tsm10/
     1  0.3596217635549736D-02,0.1093999570930850D-01,
     2  0.1879224503089045D-01,0.2771967562929827D-01,
     3  0.3896310277718949D-01/
c        
c        Weights:
c        
      data wsm10/
     1  0.7216835636822310D-02,0.7527990401385524D-02,
     2  0.8265608263833459D-02,0.9768667586014223D-02,
     3  0.1333544266090415D-01/
c        
c        
c  Data for   6 nodes
c        
c        Nodes:
c        
      data tsm9/
     1  0.4492294948184572D-02,0.1574091024493724D-01,
     2  0.2920349354749182D-01,0.4428050707464427D-01,
     3  0.6171593266575429D-01,0.8378414573843336D-01/
c        
c        Weights:
c        
      data wsm9/
     1  0.9599344134774189D-02,0.1261342607354450D-01,
     2  0.1423178130101804D-01,0.1604615663865811D-01,
     3  0.1915170695564357D-01,0.2617764436380127D-01/
c        
c        
c  Data for   6 nodes
c        
c        Nodes:
c        
      data tsm8/
     1  0.9003909925028739D-02,0.3149651793784214D-01,
     2  0.5840379766669558D-01,0.8854167709404698D-01,
     3  0.1233952856998581D+00,0.1675083333502664D+00/
c        
c        Weights:
c        
      data wsm8/
     1  0.1922225674663972D-01,0.2520864648693068D-01,
     2  0.2844682940936434D-01,0.3207621319220588D-01,
     3  0.3828429707444263D-01,0.5232540219698462D-01/
c        
c        
c  Data for   6 nodes
c        
c        Nodes:
c        
      data tsm7/
     1  0.1787549150359237D-01,0.6287445048889038D-01,
     2  0.1167963987251487D+00,0.1771956448778340D+00,
     3  0.2470819486711598D+00,0.3355855240650475D+00/
c        
c        Weights:
c        
      data wsm7/
     1  0.3827683483673964D-01,0.5051973294378791D-01,
     2  0.5700384049486597D-01,0.6429638312185580D-01,
     3  0.7679002165025135D-01,0.1049914938846487D+00/
c        
c        
c  Data for   5 nodes
c        
c        Nodes:
c        
      data tsm6/
     1  0.5749840939890877D-01,0.1749751609503663D+00,
     2  0.3007909153591551D+00,0.4442222443349930D+00,
     3  0.6253359006119926D+00/
c        
c        Weights:
c        
      data wsm6/
     1  0.1153964775339827D+00,0.1204967059922532D+00,
     2  0.1325960820476441D+00,0.1571761988120412D+00,
     3  0.2149408436007217D+00/
c        
c        
c  Data for   6 nodes
c        
c        Nodes:
c        
      data tsm5/
     1  0.7424932579676462D-01,0.2543404238445458D+00,
     2  0.4688532921320844D+00,0.7106709696869440D+00,
     3  0.9922512556855000D+00,0.1349920516519950D+01/
c        
c        Weights:
c        
      data wsm5/
     1  0.1567227617701112D+00,0.2005383370668855D+00,
     2  0.2274427358462024D+00,0.2582673081603769D+00,
     3  0.3101391543455932D+00,0.4239720353496724D+00/
c        
c        
c  Data for   7 nodes
c        
c        Nodes:
c        
      data tsm4/
     1  0.1941516865268995D+00,0.5869946012036136D+00,
     2  0.9942673438956499D+00,0.1428510313496822D+01,
     3  0.1907588639169372D+01,0.2460537480633752D+01,
     4  0.3151896128860611D+01/
c        
c        Weights:
c        
      data wsm4/
     1  0.3890469310436494D+00,0.3982577868850626D+00,
     2  0.4183365419033185D+00,0.4530488244140477D+00,
     3  0.5096867696373104D+00,0.6053857108200550D+00,
     4  0.8111289322877079D+00/
c        
c        
c  Data for   8 nodes
c        
c        Nodes:
c        
      data tsm3/
     1  0.3484470318409909D+00,0.1051524278980142D+01,
     2  0.1774010948002382D+01,0.2531847230304757D+01,
     3  0.3345937962724794D+01,0.4245506770966009D+01,
     4  0.5276791037133907D+01,0.6543108645418777D+01/
c        
c        Weights:
c        
      data wsm3/
     1  0.6979106025141834D+00,0.7104179285313381D+00,
     2  0.7371754558684469D+00,0.7819316096731849D+00,
     3  0.8510108602251312D+00,0.9554013689662911D+00,
     4  0.1121786144370488D+01,0.1465681582493974D+01/
c        
c        
c  Data for  11 nodes
c        
c        Nodes:
c        
      data tsm2/
     1  0.5720179515121766D+00,0.1721434583934795D+01,
     2  0.2887321363705357D+01,0.4081735894859774D+01,
     3  0.5318282885962752D+01,0.6612594168970237D+01,
     4  0.7982869768065220D+01,0.9451257512969768D+01,
     5  0.1104819008723231D+02,0.1282585151906421D+02,
     6  0.1491167207785179D+02/
c        
c        Weights:
c        
      data wsm2/
     1  0.1144927196540673D+01,0.1155743526152960D+01,
     2  0.1178029828307221D+01,0.1213058890541206D+01,
     3  0.1262632993446694D+01,0.1329004374587255D+01,
     4  0.1415176857220901D+01,0.1526487164149429D+01,
     5  0.1675466088556544D+01,0.1897867520712056D+01,
     6  0.2343898629049104D+01/
c        
c        
c  Data for  16 nodes
c        
c        Nodes:
c        
      data tsm1/
     1  0.8445009272013370D+00,0.2536745025663884D+01,
     2  0.4238780654973351D+01,0.5957341843503830D+01,
     3  0.7699436816960684D+01,0.9472384653756571D+01,
     4  0.1128378492050089D+02,0.1314143644098501D+02,
     5  0.1505327714749555D+02,0.1702747927925166D+02,
     6  0.1907287784029684D+02,0.2119995342137134D+02,
     7  0.2342274998064905D+02,0.2576271388289517D+02,
     8  0.2825771639086059D+02,0.3098995952689054D+02/
c        
c        Weights:
c        
      data wsm1/
     1  0.1689541126270125D+01,0.1696036570322589D+01,
     2  0.1709155479003664D+01,0.1729134281724109D+01,
     3  0.1756276044654973D+01,0.1790887746433057D+01,
     4  0.1833213927753383D+01,0.1883410956534458D+01,
     5  0.1941625876813162D+01,0.2008235373436399D+01,
     6  0.2084278831086180D+01,0.2172156672378901D+01,
     7  0.2276893424404431D+01,0.2409077873093420D+01,
     8  0.2594430057672972D+01,0.2936872166732828D+01/
c        
c        
c  Data for  26 nodes
c        
c        Nodes:
c        
      data ts0/
     1  0.1101802502658617D+01,0.3306596427404285D+01,
     2  0.5514960695890992D+01,0.7729286608702567D+01,
     3  0.9951978978813704D+01,0.1218545533662670D+02,
     4  0.1443213906545175D+02,0.1669444530977402D+02,
     5  0.1897475954339210D+02,0.2127541040326822D+02,
     6  0.2359864060609642D+02,0.2594658189869195D+02,
     7  0.2832124125707998D+02,0.3072450534744323D+02,
     8  0.3315816863567798D+02,0.3562398827806058D+02,
     9  0.3812376730308073D+02,0.4065946788189998D+02,
     *  0.4323335979214586D+02,0.4584821680330108D+02,
     1  0.4850758820662071D+02,0.5121620019382198D+02,
     2  0.5398059601214531D+02,0.5681021860148238D+02,
     3  0.5971902900263856D+02,0.6272125328241443D+02/
c        
c        Weights:
c        
      data ws0/
     1  0.2203803096735015D+01,0.2206181547797450D+01,
     2  0.2210945474494886D+01,0.2218107162704321D+01,
     3  0.2227680611069277D+01,0.2239676236840615D+01,
     4  0.2254094107081862D+01,0.2270916424873423D+01,
     5  0.2290100652100435D+01,0.2311575296663720D+01,
     6  0.2335240661052984D+01,0.2360976383923746D+01,
     7  0.2388656363697699D+01,0.2418170077742033D+01,
     8  0.2449448173203445D+01,0.2482490174903072D+01,
     9  0.2517393471310743D+01,0.2554385282624032D+01,
     *  0.2593863101364394D+01,0.2636455166915316D+01,
     1  0.2683125105474835D+01,0.2735379927829413D+01,
     2  0.2795770357873490D+01,0.2869519050142299D+01,
     3  0.2972895157000084D+01,0.3230615844900357D+01/
c        

c        

      if (ilev.ge.nlevels) goto 2000
      
      npw2=npw/2
      
      if (ilev.eq.0) then
         do i=1,npw2
            ts(i+npw2)=ts0(i)
            ws(i+npw2)=ws0(i)
         enddo
      elseif (ilev.eq.-1) then
         do i=1,npw2
            ts(i+npw2)=tsm1(i)
            ws(i+npw2)=wsm1(i)
         enddo
      elseif (ilev.eq.-2) then
         do i=1,npw2
            ts(i+npw2)=tsm2(i)
            ws(i+npw2)=wsm2(i)
         enddo
      elseif (ilev.eq.-3) then
         do i=1,npw2
            ts(i+npw2)=tsm3(i)
            ws(i+npw2)=wsm3(i)
         enddo
      elseif (ilev.eq.-4) then
         do i=1,npw2
            ts(i+npw2)=tsm4(i)
            ws(i+npw2)=wsm4(i)
         enddo
      elseif (ilev.eq.-5) then
         do i=1,npw2
            ts(i+npw2)=tsm5(i)
            ws(i+npw2)=wsm5(i)
         enddo
      elseif (ilev.eq.-6) then
         do i=1,npw2
            ts(i+npw2)=tsm6(i)
            ws(i+npw2)=wsm6(i)
         enddo
      elseif (ilev.eq.-7) then
         do i=1,npw2
            ts(i+npw2)=tsm7(i)
            ws(i+npw2)=wsm7(i)
         enddo
      elseif (ilev.eq.-8) then
         do i=1,npw2
            ts(i+npw2)=tsm8(i)
            ws(i+npw2)=wsm8(i)
         enddo
      elseif (ilev.eq.-9) then
         do i=1,npw2
            ts(i+npw2)=tsm9(i)
            ws(i+npw2)=wsm9(i)
         enddo
      elseif (ilev.eq.-10) then
         do i=1,npw2
            ts(i+npw2)=tsm10(i)
            ws(i+npw2)=wsm10(i)
         enddo
      elseif (ilev.gt.0) then
         do i=1,npw2
            ts(i+npw2)=ts0(i)*2**ilev
            ws(i+npw2)=ws0(i)*2**ilev
         enddo
      endif
      
      do i=1,npw2
         ts(npw2-i+1)=-ts(i+npw2)
         ws(npw2-i+1)=ws(i+npw2)
      enddo
c
      sqrtpi=sqrt(4*atan(1.0d0))
      do i=1,npw
         ws(i)=ws(i)/2/sqrtpi
      enddo

 2000 continue
         
      return
      end 
c
c
c
c
      subroutine get_pwnodes_md8(nlevels,ilev,npw,ws,ts)
C
C     Get planewave expansion weights,nodes for many deltas at the same level ilev
c     8 digits of accuracy
c     data npw0/6,6,6,6,6,7,7,9,12,18,29/
      
      implicit real *8 (a-h,o-z)
      real *8 ws(*),ts(*)
      real *8 wsm10(6),tsm10(6)
      real *8 wsm9(6),tsm9(6)
      real *8 wsm8(6),tsm8(6)
      real *8 wsm7(6),tsm7(6)
      real *8 wsm6(6),tsm6(6)
      real *8 wsm5(7),tsm5(7)
      real *8 wsm4(7),tsm4(7)
      real *8 wsm3(9),tsm3(9)
      real *8 wsm2(12),tsm2(12)
      real *8 wsm1(18),tsm1(18)
      real *8 ws0(29),ts0(29)
c        
c  Data for   6 nodes
c        
c        Nodes:
c        
      data tsm10/
     1  0.3521812681802748D-02,0.1066757592880692D-01,
     2  0.1814484511949723D-01,0.2627229859731333D-01,
     3  0.3558753046396872D-01,0.4731020372651179D-01/
c        
c        Weights:
c        
      data wsm10/
     1  0.7060258324574477D-02,0.7268514778719669D-02,
     2  0.7737308683760154D-02,0.8602025679981950D-02,
     3  0.1020312234375313D-01,0.1386923893797166D-01/
c        
c        
c  Data for   6 nodes
c        
c        Nodes:
c        
      data tsm9/
     1  0.7042629856711041D-02,0.2133209104515971D-01,
     2  0.3628431693948426D-01,0.5253641003921055D-01,
     3  0.7116313543176908D-01,0.9460296652018464D-01/
c        
c        Weights:
c        
      data wsm9/
     1  0.1411851363793206D-01,0.1453487405439783D-01,
     2  0.1547210685906437D-01,0.1720086643598162D-01,
     3  0.2040182731464727D-01,0.2773146901440704D-01/
c        
c        
c  Data for   6 nodes
c        
c        Nodes:
c        
      data tsm8/
     1  0.1408119096094379D-01,0.4265191834082459D-01,
     2  0.7254797677086040D-01,0.1050432890248951D+00,
     3  0.1422865299247015D+00,0.1891514893098595D+00/
c        
c        Weights:
c        
      data wsm8/
     1  0.2822888050896832D-01,0.2906148285938227D-01,
     2  0.3093565087989658D-01,0.3439242625197227D-01,
     3  0.4079211278697272D-01,0.5544314125927510D-01/
c        
c        
c  Data for   6 nodes
c        
c        Nodes:
c        
      data tsm7/
     1  0.2818875536115085D-01,0.8539130020355637D-01,
     2  0.1452723901597148D+00,0.2104079763785623D+00,
     3  0.2851361432232634D+00,0.3792606964227157D+00/
c        
c        Weights:
c        
      data wsm7/
     1  0.5651185171814125D-01,0.5819421329469447D-01,
     2  0.6198260573873190D-01,0.6897027135117544D-01,
     3  0.8189117559453617D-01,0.1113855527062270D+00/
c        
c        
c  Data for   6 nodes
c        
c        Nodes:
c        
      data tsm6/
     1  0.5627948579151301D-01,0.1705125443641418D+00,
     2  0.2901823828291242D+00,0.4205142224239513D+00,
     3  0.5702571081084599D+00,0.7590056435779232D+00/
c        
c        Weights:
c        
      data wsm6/
     1  0.1128315226681898D+00,0.1162456088359284D+00,
     2  0.1239357500041590D+00,0.1381048614864012D+00,
     3  0.1641918803483638D+00,0.2233184094734756D+00/
c        
c        
c  Data for   7 nodes
c        
c        Nodes:
c        
      data tsm5/
     1  0.1042434147129390D+00,0.3150629686544256D+00,
     2  0.5333029635662130D+00,0.7654658994822363D+00,
     3  0.1021036985370334D+01,0.1316000692072099D+01,
     4  0.1686702161283372D+01/
c        
c        Weights:
c        
      data wsm5/
     1  0.2088688994546543D+00,0.2136024579811490D+00,
     2  0.2239373078833495D+00,0.2419212844745244D+00,
     3  0.2717339362091579D+00,0.3233684663143656D+00,
     4  0.4370457242484113D+00/
c        
c        
c  Data for   7 nodes
c        
c        Nodes:
c        
      data tsm4/
     1  0.2065151403473450D+00,0.6243510852619175D+00,
     2  0.1057468536478546D+01,0.1519172637320161D+01,
     3  0.2028462427336504D+01,0.2616251667934811D+01,
     4  0.3351126708782512D+01/
c        
c        Weights:
c        
      data wsm4/
     1  0.4138173484821581D+00,0.4235690214833870D+00,
     2  0.4448391852786711D+00,0.4816521691076646D+00,
     3  0.5418042973412194D+00,0.6435312316909853D+00,
     4  0.8620556667736130D+00/
c        
c        
c  Data for   9 nodes
c        
c        Nodes:
c        
      data tsm3/
     1  0.3557927208609027D+00,0.1072436205592655D+01,
     2  0.1804780071219100D+01,0.2565099672759654D+01,
     3  0.3368574247854362D+01,0.4234969390704611D+01,
     4  0.5191885770735532D+01,0.6284650049367221D+01,
     5  0.7620201265858739D+01/
c        
c        Weights:
c        
      data wsm3/
     1  0.7124198129108908D+00,0.7226226710943214D+00,
     2  0.7440901338388681D+00,0.7790527865659936D+00,
     3  0.8311451676760644D+00,0.9061171434283236D+00,
     4  0.1014786343032102D+01,0.1185781638053005D+01,
     5  0.1543444721834471D+01/
c        
c        
c  Data for  12 nodes
c        
c        Nodes:
c        
      data tsm2/
     1  0.5817871661065473D+00,0.1749876229471543D+01,
     2  0.2931740031716246D+01,0.4137323588050713D+01,
     3  0.5377683500632913D+01,0.6665359106456457D+01,
     4  0.8014721068914692D+01,0.9442568795443119D+01,
     5  0.1096978103740030D+02,0.1262596910383255D+02,
     6  0.1446300542474593D+02,0.1660722931148534D+02/
c        
c        Weights:
c        
      data wsm2/
     1  0.1164322933364762D+01,0.1173390905788044D+01,
     2  0.1191987004217830D+01,0.1221016696395409D+01,
     3  0.1261787738564508D+01,0.1315950374068346D+01,
     4  0.1385549944290180D+01,0.1473558017590966D+01,
     5  0.1585616166711081D+01,0.1734805379302868D+01,
     6  0.1957227737847424D+01,0.2399581128859230D+01/
c        
c        
c  Data for  18 nodes
c        
c        Nodes:
c        
      data tsm1/
     1  0.8501346834327215D+00,0.2552973042611348D+01,
     2  0.4263559288776094D+01,0.5987190582008166D+01,
     3  0.7729345652344871D+01,0.9495723346612662D+01,
     4  0.1129224688241983D+02,0.1312503225231402D+02,
     5  0.1500033793717833D+02,0.1692454108384007D+02,
     6  0.1890421353199488D+02,0.2094639088164033D+02,
     7  0.2305914776901801D+02,0.2525265501270258D+02,
     8  0.2754110370570719D+02,0.2994654220527780D+02,
     9  0.3250804851804502D+02,0.3531089909425229D+02/
c        
c        Weights:
c        
      data wsm1/
     1  0.1700696840104368D+01,0.1705841715611305D+01,
     2  0.1716212555421125D+01,0.1731962484898327D+01,
     3  0.1753297384356719D+01,0.1780445770287501D+01,
     4  0.1813621579089510D+01,0.1852992817929285D+01,
     5  0.1898680368481167D+01,0.1950817007955634D+01,
     6  0.2009691285330174D+01,0.2075992402617803D+01,
     7  0.2151186712622764D+01,0.2238136381329381D+01,
     8  0.2342314095061185D+01,0.2474780225297295D+01,
     9  0.2662018412188661D+01,0.3011201803076915D+01/
c        
c        
c  Data for  29 nodes
c        
c        Nodes:
c        
      data ts0/
     1  0.1112401782179221D+01,0.3338176009124415D+01,
     2  0.5566864444476952D+01,0.7800416817243651D+01,
     3  0.1004079156060944D+02,0.1228995592770819D+02,
     4  0.1454988278766176D+02,0.1682254336132246D+02,
     5  0.1910989554550236D+02,0.2141386815196449D+02,
     6  0.2373634235140441D+02,0.2607913273020416D+02,
     7  0.2844397136653015D+02,0.3083249888066713D+02,
     8  0.3324626628028203D+02,0.3568675062657783D+02,
     9  0.3815538641155776D+02,0.4065361357591397D+02,
     *  0.4318294285395358D+02,0.4574504006899461D+02,
     1  0.4834183356422793D+02,0.5097565385799330D+02,
     2  0.5364942348467445D+02,0.5636693174473810D+02,
     3  0.5913326356639769D+02,0.6195552996346015D+02,
     4  0.6484424386195560D+02,0.6781628636359156D+02,
     *  0.7090784335888823D+02/
c        
c        Weights:
c        
      data ws0/
     1  0.2224965303311760D+01,0.2226907006555264D+01,
     2  0.2230794774667794D+01,0.2236636372263983D+01,
     3  0.2244441041583004D+01,0.2254216624331466D+01,
     4  0.2265965827780426D+01,0.2279681901613941D+01,
     5  0.2295344250846016D+01,0.2312914815274751D+01,
     6  0.2332336291020476D+01,0.2353533306988707D+01,
     7  0.2376417382865340D+01,0.2400895897200978D+01,
     8  0.2426884571639051D+01,0.2454322431319502D+01,
     9  0.2483188100292755D+01,0.2513516747250493D+01,
     *  0.2545417965357919D+01,0.2579096274092056D+01,
     1  0.2614877880878113D+01,0.2653250449257397D+01,
     2  0.2694928711379689D+01,0.2740973228443774D+01,
     3  0.2793031519767612D+01,0.2853924415752795D+01,
     4  0.2929566669523441D+01,0.3039245024192118D+01,
     *  0.3362417600326868D+01/
c        

c        

      if (ilev.ge.nlevels) goto 2000
      
      npw2=npw/2
      
      if (ilev.eq.0) then
         do i=1,npw2
            ts(i+npw2)=ts0(i)
            ws(i+npw2)=ws0(i)
         enddo
      elseif (ilev.eq.-1) then
         do i=1,npw2
            ts(i+npw2)=tsm1(i)
            ws(i+npw2)=wsm1(i)
         enddo
      elseif (ilev.eq.-2) then
         do i=1,npw2
            ts(i+npw2)=tsm2(i)
            ws(i+npw2)=wsm2(i)
         enddo
      elseif (ilev.eq.-3) then
         do i=1,npw2
            ts(i+npw2)=tsm3(i)
            ws(i+npw2)=wsm3(i)
         enddo
      elseif (ilev.eq.-4) then
         do i=1,npw2
            ts(i+npw2)=tsm4(i)
            ws(i+npw2)=wsm4(i)
         enddo
      elseif (ilev.eq.-5) then
         do i=1,npw2
            ts(i+npw2)=tsm5(i)
            ws(i+npw2)=wsm5(i)
         enddo
      elseif (ilev.eq.-6) then
         do i=1,npw2
            ts(i+npw2)=tsm6(i)
            ws(i+npw2)=wsm6(i)
         enddo
      elseif (ilev.eq.-7) then
         do i=1,npw2
            ts(i+npw2)=tsm7(i)
            ws(i+npw2)=wsm7(i)
         enddo
      elseif (ilev.eq.-8) then
         do i=1,npw2
            ts(i+npw2)=tsm8(i)
            ws(i+npw2)=wsm8(i)
         enddo
      elseif (ilev.eq.-9) then
         do i=1,npw2
            ts(i+npw2)=tsm9(i)
            ws(i+npw2)=wsm9(i)
         enddo
      elseif (ilev.eq.-10) then
         do i=1,npw2
            ts(i+npw2)=tsm10(i)
            ws(i+npw2)=wsm10(i)
         enddo
      elseif (ilev.gt.0) then
         do i=1,npw2
            ts(i+npw2)=ts0(i)*2**ilev
            ws(i+npw2)=ws0(i)*2**ilev
         enddo
      endif
      
      do i=1,npw2
         ts(npw2-i+1)=-ts(i+npw2)
         ws(npw2-i+1)=ws(i+npw2)
      enddo
c
      sqrtpi=sqrt(4*atan(1.0d0))
      do i=1,npw
         ws(i)=ws(i)/2/sqrtpi
      enddo

 2000 continue
         
      return
      end 
c
c
c
c
      subroutine get_pwnodes_md9(nlevels,ilev,npw,ws,ts)
C
C     Get planewave expansion weights,nodes for many deltas at the same level ilev
c     9 digits of accuracy
c     data npw0/7,7,7,7,7,7,8,10,13,20,33/
      
      implicit real *8 (a-h,o-z)
      real *8 ws(*),ts(*)
      real *8 wsm10(7),tsm10(7)
      real *8 wsm9(7),tsm9(7)
      real *8 wsm8(7),tsm8(7)
      real *8 wsm7(7),tsm7(7)
      real *8 wsm6(7),tsm6(7)
      real *8 wsm5(7),tsm5(7)
      real *8 wsm4(8),tsm4(8)
      real *8 wsm3(10),tsm3(10)
      real *8 wsm2(13),tsm2(13)
      real *8 wsm1(20),tsm1(20)
      real *8 ws0(33),ts0(33)
c        
c  Data for   7 nodes
c        
c        Nodes:
c        
      data tsm10/
     1  0.3397284187682642D-02,0.1026479940541047D-01,
     2  0.1736470564386756D-01,0.2490272421448856D-01,
     3  0.3318554837035874D-01,0.4274593389947312D-01,
     4  0.5481758768239677D-01/
c        
c        Weights:
c        
      data wsm10/
     1  0.6806512314503734D-02,0.6954579373016099D-02,
     2  0.7278688648480121D-02,0.7846727912171305D-02,
     3  0.8802451828948630D-02,0.1049426684266971D-01,
     4  0.1429275169131434D-01/
c        
c        
c  Data for   7 nodes
c        
c        Nodes:
c        
      data tsm9/
     1  0.6795237249873807D-02,0.2053148134372907D-01,
     2  0.3473211952545183D-01,0.4980827917287395D-01,
     3  0.6637293420351634D-01,0.8549101735063958D-01,
     4  0.1096289251462865D+00/
c        
c        Weights:
c        
      data wsm9/
     1  0.1361434194599990D-01,0.1391022534829530D-01,
     2  0.1455791032798801D-01,0.1569310001847683D-01,
     3  0.1760317898337511D-01,0.2098473262445688D-01,
     4  0.2857804655364470D-01/
c        
c        
c  Data for   7 nodes
c        
c        Nodes:
c        
      data tsm8/
     1  0.1358931785031237D-01,0.4105965868049104D-01,
     2  0.6945933720319869D-01,0.9961113917340644D-01,
     3  0.1327417097140779D+00,0.1709814674023832D+00,
     4  0.2192623244330889D+00/
c        
c        Weights:
c        
      data wsm8/
     1  0.2722639738817212D-01,0.2781850470866729D-01,
     2  0.2911465860193731D-01,0.3138644752860303D-01,
     3  0.3520874135941315D-01,0.4197417888143834D-01,
     4  0.5716046281638666D-01/
c        
c        
c  Data for   7 nodes
c        
c        Nodes:
c        
      data tsm7/
     1  0.2774791492634180D-01,0.8383870305800865D-01,
     2  0.1418211944476062D+00,0.2033551514156430D+00,
     3  0.2708886154207371D+00,0.3486425856754114D+00,
     4  0.4464122060787699D+00/
c        
c        Weights:
c        
      data wsm7/
     1  0.5559329910061313D-01,0.5680038081963507D-01,
     2  0.5943478334221763D-01,0.6402756809366923D-01,
     3  0.7170083866161941D-01,0.8520135944510519D-01,
     4  0.1154777151953363D+00/
c        
c        
c  Data for   7 nodes
c        
c        Nodes:
c        
      data tsm6/
     1  0.5509528897274551D-01,0.1664799414224010D+00,
     2  0.2816663124983253D+00,0.4040074740696331D+00,
     3  0.5384669204571864D+00,0.6935836020827906D+00,
     4  0.8890027176929657D+00/
c        
c        Weights:
c        
      data wsm6/
     1  0.1103861178713862D+00,0.1128095309718160D+00,
     2  0.1181088844472938D+00,0.1273734969097302D+00,
     3  0.1428855423650030D+00,0.1701487979204437D+00,
     4  0.2309409433772552D+00/
c        
c        
c  Data for   7 nodes
c        
c        Nodes:
c        
      data tsm5/
     1  0.1105979697897531D+00,0.3342803493809506D+00,
     2  0.5658717686071139D+00,0.8122986196553983D+00,
     3  0.1083645567422562D+01,0.1396855501748587D+01,
     4  0.1790366563936449D+01/
c        
c        Weights:
c        
      data wsm5/
     1  0.2216031866409320D+00,0.2266486441976390D+00,
     2  0.2376632747184920D+00,0.2568206177046607D+00,
     3  0.2885378771014901D+00,0.3433547248712815D+00,
     4  0.4637705569733719D+00/
c        
c        
c  Data for   8 nodes
c        
c        Nodes:
c        
      data tsm4/
     1  0.2051497624943500D+00,0.6190747793742033D+00,
     2  0.1044372917985098D+01,0.1490390897403198D+01,
     3  0.1969555372674927D+01,0.2499883812172329D+01,
     4  0.3111351746991225D+01,0.3873181735758759D+01/
c        
c        Weights:
c        
      data wsm4/
     1  0.4108957453759666D+00,0.4182277515944975D+00,
     2  0.4339027386761018D+00,0.4601668901672795D+00,
     3  0.5010810402879307D+00,0.5642755810665019D+00,
     4  0.6684437355478672D+00,0.8922465702289711D+00/
c        
c        
c  Data for  10 nodes
c        
c        Nodes:
c        
      data tsm3/
     1  0.3584295084354250D+00,0.1079413942413030D+01,
     2  0.1813123321116114D+01,0.2569234941378820D+01,
     3  0.3359326694257502D+01,0.4197890205393017D+01,
     4  0.5103886846287842D+01,0.6104242514659085D+01,
     5  0.7244608090680696D+01,0.8635621318746162D+01/
c        
c        Weights:
c        
      data wsm3/
     1  0.7175408525678086D+00,0.7258492610656502D+00,
     2  0.7431688309392634D+00,0.7709677589049700D+00,
     3  0.8116076318161599D+00,0.8686306435244939D+00,
     4  0.9477083672197123D+00,0.1060098049352603D+01,
     5  0.1236164010502727D+01,0.1606498357343857D+01/
c        
c        
c  Data for  13 nodes
c        
c        Nodes:
c        
      data tsm2/
     1  0.5918068481561684D+00,0.1779326506262149D+01,
     2  0.2978733205957081D+01,0.4198511308998243D+01,
     3  0.5447965253093822D+01,0.6737499033279241D+01,
     4  0.8078858067021203D+01,0.9485433676055710D+01,
     5  0.1097297094339953D+02,0.1256147229515313D+02,
     6  0.1428013870951218D+02,0.1618115810143876D+02,
     7  0.1839248604646501D+02/
c        
c        Weights:
c        
      data wsm2/
     1  0.1184261863017795D+01,0.1192102035383556D+01,
     2  0.1208120088681173D+01,0.1232982166820491D+01,
     3  0.1267655870174428D+01,0.1313364924442699D+01,
     4  0.1371574965927884D+01,0.1444172387788406D+01,
     5  0.1534163547989795D+01,0.1647516541222239D+01,
     6  0.1797908206058326D+01,0.2022322366000357D+01,
     7  0.2469609903345974D+01/
c        
c        
c  Data for  20 nodes
c        
c        Nodes:
c        
      data tsm1/
     1  0.8546467874834279D+00,0.2566026479308387D+01,
     2  0.4283691594800382D+01,0.6011920403195300D+01,
     3  0.7755114958404898D+01,0.9517833422282092D+01,
     4  0.1130480439571911D+02,0.1312091985145755D+02,
     5  0.1497120941335622D+02,0.1686080990743885D+02,
     6  0.1879495867138569D+02,0.2077905282433896D+02,
     7  0.2281882647262390D+02,0.2492070787512353D+02,
     8  0.2709244601699700D+02,0.2934417910096465D+02,
     9  0.3169035003823619D+02,0.3415356104435018D+02,
     *  0.3677391732875611D+02,0.3963935021109020D+02/
c        
c        Weights:
c        
      data wsm1/
     1  0.1709640804653967D+01,0.1713817587362028D+01,
     2  0.1722224928359197D+01,0.1734965811470756D+01,
     3  0.1752182743964191D+01,0.1774042406063983D+01,
     4  0.1800715329866326D+01,0.1832354427983204D+01,
     5  0.1869081047706949D+01,0.1910991967114548D+01,
     6  0.1958202270062765D+01,0.2010935963704395D+01,
     7  0.2069672636560532D+01,0.2135364718922703D+01,
     8  0.2209771627462439D+01,0.2296044675176862D+01,
     9  0.2399938729390543D+01,0.2532848900787195D+01,
     *  0.2721897689677442D+01,0.3077049613409732D+01/
c        
c        
c  Data for  33 nodes
c        
c        Nodes:
c        
      data ts0/
     1  0.1113661467122586D+01,0.3341752759900283D+01,
     2  0.5572150562110177D+01,0.7806397099147069D+01,
     3  0.1004604062812116D+02,0.1229263625221200D+02,
     4  0.1454774514211204D+02,0.1681293175804197D+02,
     5  0.1908975877249787D+02,0.2137977959455052D+02,
     6  0.2368452871260419D+02,0.2600551049979521D+02,
     7  0.2834418761936337D+02,0.3070197063623727D+02,
     8  0.3308021076928356D+02,0.3548019780508980D+02,
     9  0.3790316500266565D+02,0.4035030240972250D+02,
     *  0.4282277953734776D+02,0.4532177800543107D+02,
     1  0.4784853478631750D+02,0.5040439723443588D+02,
     2  0.5299089240955024D+02,0.5560981562196507D+02,
     3  0.5826334734693668D+02,0.6095421521264687D+02,
     4  0.6368593223783394D+02,0.6646317275598859D+02,
     5  0.6929241782213195D+02,0.7218318984732053D+02,
     6  0.7515080814578288D+02,0.7822489440064369D+02,
     *  0.8161142117900535D+02/
c        
c        Weights:
c        
      data ws0/
     1  0.2227450969564163D+01,0.2228987929377683D+01,
     2  0.2232064678889539D+01,0.2236686423379106D+01,
     3  0.2242859830415982D+01,0.2250591658429895D+01,
     4  0.2259886949079065D+01,0.2270746853264119D+01,
     5  0.2283166239622572D+01,0.2297131341850088D+01,
     6  0.2312617820251489D+01,0.2329589709206378D+01,
     7  0.2347999749847858D+01,0.2367791526329223D+01,
     8  0.2388903625978056D+01,0.2411275769276593D+01,
     9  0.2434856590974877D+01,0.2459612599645233D+01,
     *  0.2485537875710643D+01,0.2512664314458922D+01,
     1  0.2541072670046590D+01,0.2570905299671730D+01,
     2  0.2602382403771184D+01,0.2635824922741431D+01,
     3  0.2671689626363682D+01,0.2710626592734448D+01,
     4  0.2753579432224739D+01,0.2801973162917049D+01,
     5  0.2858101057112886D+01,0.2926036196619778D+01,
     6  0.3014253573422082D+01,0.3147522861006978D+01,
     *  0.4152663315480152D+01/
c        

c        

      if (ilev.ge.nlevels) goto 2000
      
      npw2=npw/2
      
      if (ilev.eq.0) then
         do i=1,npw2
            ts(i+npw2)=ts0(i)
            ws(i+npw2)=ws0(i)
         enddo
      elseif (ilev.eq.-1) then
         do i=1,npw2
            ts(i+npw2)=tsm1(i)
            ws(i+npw2)=wsm1(i)
         enddo
      elseif (ilev.eq.-2) then
         do i=1,npw2
            ts(i+npw2)=tsm2(i)
            ws(i+npw2)=wsm2(i)
         enddo
      elseif (ilev.eq.-3) then
         do i=1,npw2
            ts(i+npw2)=tsm3(i)
            ws(i+npw2)=wsm3(i)
         enddo
      elseif (ilev.eq.-4) then
         do i=1,npw2
            ts(i+npw2)=tsm4(i)
            ws(i+npw2)=wsm4(i)
         enddo
      elseif (ilev.eq.-5) then
         do i=1,npw2
            ts(i+npw2)=tsm5(i)
            ws(i+npw2)=wsm5(i)
         enddo
      elseif (ilev.eq.-6) then
         do i=1,npw2
            ts(i+npw2)=tsm6(i)
            ws(i+npw2)=wsm6(i)
         enddo
      elseif (ilev.eq.-7) then
         do i=1,npw2
            ts(i+npw2)=tsm7(i)
            ws(i+npw2)=wsm7(i)
         enddo
      elseif (ilev.eq.-8) then
         do i=1,npw2
            ts(i+npw2)=tsm8(i)
            ws(i+npw2)=wsm8(i)
         enddo
      elseif (ilev.eq.-9) then
         do i=1,npw2
            ts(i+npw2)=tsm9(i)
            ws(i+npw2)=wsm9(i)
         enddo
      elseif (ilev.eq.-10) then
         do i=1,npw2
            ts(i+npw2)=tsm10(i)
            ws(i+npw2)=wsm10(i)
         enddo
      elseif (ilev.gt.0) then
         do i=1,npw2
            ts(i+npw2)=ts0(i)*2**ilev
            ws(i+npw2)=ws0(i)*2**ilev
         enddo
      endif
      
      do i=1,npw2
         ts(npw2-i+1)=-ts(i+npw2)
         ws(npw2-i+1)=ws(i+npw2)
      enddo
c
      sqrtpi=sqrt(4*atan(1.0d0))
      do i=1,npw
         ws(i)=ws(i)/2/sqrtpi
      enddo

 2000 continue
         
      return
      end 
c
c
c
c
      subroutine get_pwnodes_md10(nlevels,ilev,npw,ws,ts)
C
C     Get planewave expansion weights,nodes for many deltas at the same level ilev
c     10 digits of accuracy
c     data npw0/7,7,8,8,8,8,9,11,15,22,36/
      
      implicit real *8 (a-h,o-z)
      real *8 ws(*),ts(*)
      real *8 wsm10(7),tsm10(7)
      real *8 wsm9(7),tsm9(7)
      real *8 wsm8(8),tsm8(8)
      real *8 wsm7(8),tsm7(8)
      real *8 wsm6(8),tsm6(8)
      real *8 wsm5(8),tsm5(8)
      real *8 wsm4(9),tsm4(9)
      real *8 wsm3(11),tsm3(11)
      real *8 wsm2(15),tsm2(15)
      real *8 wsm1(22),tsm1(22)
      real *8 ws0(36),ts0(36)
c        
c  Data for   7 nodes
c        
c        Nodes:
c        
      data tsm10/
     1  0.3654685526997541D-02,0.1104161578002788D-01,
     2  0.1867508236708187D-01,0.2677122906192292D-01,
     3  0.3564898990410208D-01,0.4585963331861174D-01,
     4  0.5868689616006234D-01/
c        
c        Weights:
c        
      data wsm10/
     1  0.7322078403339444D-02,0.7479422741047173D-02,
     2  0.7822692527259822D-02,0.8420971844607788D-02,
     3  0.9420891918078296D-02,0.1118304229533848D-01,
     4  0.1514623458289506D-01/
c        
c        
c  Data for   7 nodes
c        
c        Nodes:
c        
      data tsm9/
     1  0.7308231844131724D-02,0.2207975974554838D-01,
     2  0.3734418488772951D-01,0.5353363075128263D-01,
     3  0.7128571412746824D-01,0.9170254596958431D-01,
     4  0.1173506784019955D+00/
c        
c        Weights:
c        
      data wsm9/
     1  0.1464186956099839D-01,0.1495644778915246D-01,
     2  0.1564274003686110D-01,0.1683885300414098D-01,
     3  0.1883793040032654D-01,0.2236089489496629D-01,
     4  0.3028441441860252D-01/
c        
c        
c  Data for   8 nodes
c        
c        Nodes:
c        
      data tsm8/
     1  0.1388183964089923D-01,0.4188035792617861D-01,
     2  0.7061561165065278D-01,0.1006954578034097D+00,
     3  0.1329413547226420D+00,0.1685862002842833D+00,
     4  0.2097701519470205D+00,0.2615474627492719D+00/
c        
c        Weights:
c        
      data wsm8/
     1  0.2780230252089974D-01,0.2827720605344673D-01,
     2  0.2929292211426194D-01,0.3100072046300720D-01,
     3  0.3368984675503731D-01,0.3793444508006114D-01,
     4  0.4514911649373473D-01,0.6110302638145701D-01/
c        
c        
c  Data for   8 nodes
c        
c        Nodes:
c        
      data tsm7/
     1  0.2775577712270937D-01,0.8374011486061643D-01,
     2  0.1412077887568243D+00,0.2013828018935612D+00,
     3  0.2659181926815564D+00,0.3372897621728430D+00,
     4  0.4197803641145989D+00,0.5234722942627514D+00/
c        
c        Weights:
c        
      data wsm7/
     1  0.5558930899774873D-01,0.5654541973463117D-01,
     2  0.5859054244976846D-01,0.6202909642650224D-01,
     3  0.6744100972315575D-01,0.7597203827939564D-01,
     4  0.9043713664976527D-01,0.1223217645701759D+00/
c        
c        
c  Data for   8 nodes
c        
c        Nodes:
c        
      data tsm6/
     1  0.3669361772437100D-01,0.1290373442758702D+00,
     2  0.2379373938294360D+00,0.3554793929020926D+00,
     3  0.4824535366451594D+00,0.6231451747186840D+00,
     4  0.7859188235678510D+00,0.9907589572727620D+00/
c        
c        Weights:
c        
      data wsm6/
     1  0.7865326490903737D-01,0.1031967256213944D+00,
     2  0.1134971209830649D+00,0.1217892751920897D+00,
     3  0.1328621402966455D+00,0.1498328524286513D+00,
     4  0.1785506213784820D+00,0.2417454676449271D+00/
c        
c        
c  Data for   8 nodes
c        
c        Nodes:
c        
      data tsm5/
     1  0.1093244067431166D+00,0.3298528904468653D+00,
     2  0.5562801763249032D+00,0.7934742851964770D+00,
     3  0.1047989609683216D+01,0.1329560759804355D+01,
     4  0.1654827269441886D+01,0.2062654314911940D+01/
c        
c        Weights:
c        
      data wsm5/
     1  0.2189579162806903D+00,0.2227594656640628D+00,
     2  0.2308932313333243D+00,0.2445650021832178D+00,
     3  0.2660380153253792D+00,0.2997098780481433D+00,
     4  0.3563265752083541D+00,0.4800393024298774D+00/
c        
c        
c  Data for   9 nodes
c        
c        Nodes:
c        
      data tsm4/
     1  0.2043116347406652D+00,0.6157869116407541D+00,
     2  0.1036127542860857D+01,0.1472330760636751D+01,
     3  0.1933237411312885D+01,0.2430813281219760D+01,
     4  0.2982598731940782D+01,0.3618437942017911D+01,
     5  0.4409268210376500D+01/
c        
c        Weights:
c        
      data wsm4/
     1  0.4090935770243928D+00,0.4148482438985137D+00,
     2  0.4269853493469833D+00,0.4468745012766201D+00,
     3  0.4768973814987302D+00,0.5210888894240103D+00,
     4  0.5871403304288778D+00,0.6945292707178983D+00,
     5  0.9255786010346240D+00/
c        
c        
c  Data for  11 nodes
c        
c        Nodes:
c        
      data tsm3/
     1  0.3605734149690762D+00,0.1085148136963090D+01,
     2  0.1820246300244567D+01,0.2573701805983795D+01,
     3  0.3354641653254939D+01,0.4174143249430152D+01,
     4  0.5046120657284398D+01,0.5988896000927336D+01,
     5  0.7028987946851641D+01,0.8212575833578105D+01,
     6  0.9653723888607987D+01/
c        
c        Weights:
c        
      data wsm3/
     1  0.7217141955390221D+00,0.7286096172634481D+00,
     2  0.7428832040460576D+00,0.7655382371844989D+00,
     3  0.7981737700488064D+00,0.8431251655704992D+00,
     4  0.9038296642906083D+00,0.9859879132084487D+00,
     5  0.1101355797324311D+01,0.1281811867537652D+01,
     6  0.1663358803079868D+01/
c        
c        
c  Data for  15 nodes
c        
c        Nodes:
c        
      data tsm2/
     1  0.5490597821987904D+00,0.1652888459309952D+01,
     2  0.2772754967994542D+01,0.3916741225220101D+01,
     3  0.5091252120059071D+01,0.6302297579445771D+01,
     4  0.7556518861489295D+01,0.8861753588881090D+01,
     5  0.1022734365265067D+02,0.1166451154220922D+02,
     6  0.1318725473382581D+02,0.1481453424468642D+02,
     7  0.1657556713076281D+02,0.1852425452949646D+02,
     8  0.2079457038228772D+02/
c        
c        Weights:
c        
      data wsm2/
     1  0.1099092520237628D+01,0.1110307295281710D+01,
     2  0.1130759724695677D+01,0.1158256401485009D+01,
     3  0.1191749693256775D+01,0.1231439282733500D+01,
     4  0.1278305179978544D+01,0.1333715989010650D+01,
     5  0.1399321298337010D+01,0.1477314428160626D+01,
     6  0.1571254232120176D+01,0.1687976342083156D+01,
     7  0.1842438800482014D+01,0.2073952020959631D+01,
     8  0.2539070036394666D+01/
c        
c        
c  Data for  22 nodes
c        
c        Nodes:
c        
      data tsm1/
     1  0.8583417610801847D+00,0.2576753165562916D+01,
     2  0.4300366832870928D+01,0.6032711766404955D+01,
     3  0.7777403783639314D+01,0.9538170969759012D+01,
     4  0.1131886926793126D+02,0.1312348559055146D+02,
     5  0.1495612787945023D+02,0.1682100586559585D+02,
     6  0.1872241294523807D+02,0.2066472745965500D+02,
     7  0.2265245874850662D+02,0.2469036851128545D+02,
     8  0.2678370341377374D+02,0.2893858803900583D+02,
     9  0.3116266386267236D+02,0.3346615384530145D+02,
     *  0.3586377534213780D+02,0.3837862911240652D+02,
     1  0.4105172645051724D+02,0.4397337700349861D+02/
c        
c        Weights:
c        
      data wsm1/
     1  0.1716971189429874D+01,0.1720430076756137D+01,
     2  0.1727384881588108D+01,0.1737907189793216D+01,
     3  0.1752097925532774D+01,0.1770079049733159D+01,
     4  0.1791982205575634D+01,0.1817935441552147D+01,
     5  0.1848051104048878D+01,0.1882420474387950D+01,
     6  0.1921122713933636D+01,0.1964255931947079D+01,
     7  0.2011996494579805D+01,0.2064691111712126D+01,
     8  0.2122989191664886D+01,0.2188036784296733D+01,
     9  0.2261789117377338D+01,0.2347586098623890D+01,
     *  0.2451377974613604D+01,0.2584826846016103D+01,
     1  0.2775629921042685D+01,0.3136243329310611D+01/
c        
c        
c  Data for  36 nodes
c        
c        Nodes:
c        
      data ts0/
     1  0.1116087358502530D+01,0.3348891117520576D+01,
     2  0.5583582957456391D+01,0.7821424659783522D+01,
     3  0.1006368216890879D+02,0.1231162639807185D+02,
     4  0.1456653317551284D+02,0.1682968209892203D+02,
     5  0.1910235410143558D+02,0.2138582760206129D+02,
     6  0.2368137322907248D+02,0.2599024727315056D+02,
     7  0.2831368424369474D+02,0.3065288914685644D+02,
     8  0.3300903034084794D+02,0.3538323400534532D+02,
     9  0.3777658134226346D+02,0.4019010957950937D+02,
     *  0.4262481768603252D+02,0.4508167747192658D+02,
     1  0.4756165051675972D+02,0.5006571123048589D+02,
     2  0.5259487638868775D+02,0.5515024177413921D+02,
     3  0.5773302718116817D+02,0.6034463211874539D+02,
     4  0.6298670630451100D+02,0.6566124193084606D+02,
     5  0.6837069963725297D+02,0.7111818912523161D+02,
     6  0.7390774299086924D+02,0.7674476098545721D+02,
     7  0.7963678772235379D+02,0.8259504037027968D+02,
     8  0.8563791009177559D+02,0.8880714082996738D+02/
c        
c        Weights:
c        
      data ws0/
     1  0.2232279541135866D+01,0.2233537786339483D+01,
     2  0.2236056166182051D+01,0.2239838215186893D+01,
     3  0.2244888622273500D+01,0.2251212492703534D+01,
     4  0.2258814368706445D+01,0.2267697031080010D+01,
     5  0.2277860132080369D+01,0.2289298750544367D+01,
     6  0.2302002010800591D+01,0.2315951959690175D+01,
     7  0.2331122937360099D+01,0.2347481690246390D+01,
     8  0.2364988443647418D+01,0.2383599071206264D+01,
     9  0.2403268380434199D+01,0.2423954405144105D+01,
     *  0.2445623495236307D+01,0.2468255957102927D+01,
     1  0.2491852045242791D+01,0.2516438240988789D+01,
     2  0.2542073970082657D+01,0.2568859201673218D+01,
     3  0.2596943755611409D+01,0.2626539691018870D+01,
     4  0.2657939009251882D+01,0.2691540418459413D+01,
     5  0.2727891887598084D+01,0.2767762039985074D+01,
     6  0.2812268423633963D+01,0.2863136148243723D+01,
     7  0.2923318861397995D+01,0.2999028122648916D+01,
     8  0.3110487229256403D+01,0.3444335269611202D+01/
c        

c        

      if (ilev.ge.nlevels) goto 2000
      
      npw2=npw/2
      
      if (ilev.eq.0) then
         do i=1,npw2
            ts(i+npw2)=ts0(i)
            ws(i+npw2)=ws0(i)
         enddo
      elseif (ilev.eq.-1) then
         do i=1,npw2
            ts(i+npw2)=tsm1(i)
            ws(i+npw2)=wsm1(i)
         enddo
      elseif (ilev.eq.-2) then
         do i=1,npw2
            ts(i+npw2)=tsm2(i)
            ws(i+npw2)=wsm2(i)
         enddo
      elseif (ilev.eq.-3) then
         do i=1,npw2
            ts(i+npw2)=tsm3(i)
            ws(i+npw2)=wsm3(i)
         enddo
      elseif (ilev.eq.-4) then
         do i=1,npw2
            ts(i+npw2)=tsm4(i)
            ws(i+npw2)=wsm4(i)
         enddo
      elseif (ilev.eq.-5) then
         do i=1,npw2
            ts(i+npw2)=tsm5(i)
            ws(i+npw2)=wsm5(i)
         enddo
      elseif (ilev.eq.-6) then
         do i=1,npw2
            ts(i+npw2)=tsm6(i)
            ws(i+npw2)=wsm6(i)
         enddo
      elseif (ilev.eq.-7) then
         do i=1,npw2
            ts(i+npw2)=tsm7(i)
            ws(i+npw2)=wsm7(i)
         enddo
      elseif (ilev.eq.-8) then
         do i=1,npw2
            ts(i+npw2)=tsm8(i)
            ws(i+npw2)=wsm8(i)
         enddo
      elseif (ilev.eq.-9) then
         do i=1,npw2
            ts(i+npw2)=tsm9(i)
            ws(i+npw2)=wsm9(i)
         enddo
      elseif (ilev.eq.-10) then
         do i=1,npw2
            ts(i+npw2)=tsm10(i)
            ws(i+npw2)=wsm10(i)
         enddo
      elseif (ilev.gt.0) then
         do i=1,npw2
            ts(i+npw2)=ts0(i)*2**ilev
            ws(i+npw2)=ws0(i)*2**ilev
         enddo
      endif
      
      do i=1,npw2
         ts(npw2-i+1)=-ts(i+npw2)
         ws(npw2-i+1)=ws(i+npw2)
      enddo
c
      sqrtpi=sqrt(4*atan(1.0d0))
      do i=1,npw
         ws(i)=ws(i)/2/sqrtpi
      enddo

 2000 continue
         
      return
      end 
c
c
c
c
      subroutine get_pwnodes_md11(nlevels,ilev,npw,ws,ts)
C
C     Get planewave expansion weights,nodes for many deltas at the same level ilev
c     11 digits of accuracy
c     data npw0/8,8,8,8,8,9,10,12,16,25,40/
      
      implicit real *8 (a-h,o-z)
      real *8 ws(*),ts(*)
      real *8 wsm10(8),tsm10(8)
      real *8 wsm9(8),tsm9(8)
      real *8 wsm8(8),tsm8(8)
      real *8 wsm7(8),tsm7(8)
      real *8 wsm6(8),tsm6(8)
      real *8 wsm5(9),tsm5(9)
      real *8 wsm4(10),tsm4(10)
      real *8 wsm3(12),tsm3(12)
      real *8 wsm2(16),tsm2(16)
      real *8 wsm1(25),tsm1(25)
      real *8 ws0(40),ts0(40)

c        
c        
c  Data for   8 nodes
c        
c        Nodes:
c        
      data tsm10/
     1  0.3592496062367289D-02,0.1083572411701319D-01,
     2  0.1826157536823494D-01,0.2602060939087721D-01,
     3  0.3431675181256182D-01,0.4345832399148843D-01,
     4  0.5398627400107721D-01,0.6718686488926741D-01/
c        
c        Weights:
c        
      data wsm10/
     1  0.7194571459535021D-02,0.7312325214601546D-02,
     2  0.7564045540802894D-02,0.7987247457866638D-02,
     3  0.8654680650392934D-02,0.9712951650648515D-02,
     4  0.1152512893334295D-01,0.1556252164525134D-01/
c        
c        
c  Data for   8 nodes
c        
c        Nodes:
c        
      data tsm9/
     1  0.7183881020636958D-02,0.2166809455262847D-01,
     2  0.3651749191552022D-01,0.5203314489092910D-01,
     3  0.6862284379624210D-01,0.8690313000379132D-01,
     4  0.1079556694081327D+00,0.1343523037648201D+00/
c        
c        Weights:
c        
      data wsm9/
     1  0.1438691733901489D-01,0.1462238387782063D-01,
     2  0.1512573957276300D-01,0.1597200649091376D-01,
     3  0.1730666217792165D-01,0.1942285169690878D-01,
     4  0.2304649244401244D-01,0.3111933042052841D-01/
c        
c        
c  Data for   8 nodes
c        
c        Nodes:
c        
      data tsm8/
     1  0.1437124888192139D-01,0.4334734018663324D-01,
     2  0.7305600146564119D-01,0.1041012904669711D+00,
     3  0.1373013817476967D+00,0.1738928428900725D+00,
     4  0.2160435072625569D+00,0.2689037150369928D+00/
c        
c        Weights:
c        
      data wsm8/
     1  0.2878092131008850D-01,0.2925325704459669D-01,
     2  0.3026303719011013D-01,0.3196084954904136D-01,
     3  0.3463845455180962D-01,0.3888309399500964D-01,
     4  0.4614776800338580D-01,0.6232042411030800D-01/
c        
c        
c  Data for   8 nodes
c        
c        Nodes:
c        
      data tsm7/
     1  0.2873890365480227D-01,0.8668786233862867D-01,
     2  0.1461146525295710D+00,0.2082385514293759D+00,
     3  0.2747110542267067D+00,0.3480252381125546D+00,
     4  0.4325411472927746D+00,0.5385931490458190D+00/
c        
c        Weights:
c        
      data wsm7/
     1  0.5755530197240365D-01,0.5850803404604138D-01,
     2  0.6054532253299246D-01,0.6397160090596930D-01,
     3  0.6937498375117318D-01,0.7793469008239313D-01,
     4  0.9256122084417863D-01,0.1250521546335335D+00/
c        
c        
c  Data for   8 nodes
c        
c        Nodes:
c        
      data tsm6/
     1  0.5749589682326198D-01,0.1734517243315231D+00,
     2  0.2924322890896927D+00,0.4169346058301294D+00,
     3  0.5503350900936130D+00,0.6977011614673962D+00,
     4  0.8678152802229542D+00,0.1081353939948275D+01/
c        
c        Weights:
c        
      data wsm6/
     1  0.1151503395861952D+00,0.1170999173327803D+00,
     2  0.1212704254321419D+00,0.1282847868990791D+00,
     3  0.1393333393298478D+00,0.1567725474585165D+00,
     4  0.1863851576423720D+00,0.2517046308351242D+00/
c        
c        
c  Data for   9 nodes
c        
c        Nodes:
c        
      data tsm5/
     1  0.7944424159233136D-01,0.2627479418319628D+00,
     2  0.4732685858569344D+00,0.6996814807791654D+00,
     3  0.9414497273570835D+00,0.1203369636223478D+01,
     4  0.1494529599630674D+01,0.1831463903976633D+01,
     5  0.2253880063572655D+01/
c        
c        Weights:
c        
      data wsm5/
     1  0.1649851437377325D+00,0.1999041024582673D+00,
     2  0.2192204680717173D+00,0.2336280036755917D+00,
     3  0.2507183075856640D+00,0.2745939263540540D+00,
     4  0.3103097746240256D+00,0.3691663293073131D+00,
     5  0.4970621067205222D+00/
c        
c        
c  Data for  10 nodes
c        
c        Nodes:
c        
      data tsm4/
     1  0.1872540544162017D+00,0.5666961649223257D+00,
     2  0.9592436192028414D+00,0.1370530013372342D+01,
     3  0.1805811942811487D+01,0.2272199614512827D+01,
     4  0.2780351152640172D+01,0.3346941180710977D+01,
     5  0.4001542964291191D+01,0.4816371706654675D+01/
c        
c        Weights:
c        
      data wsm4/
     1  0.3753686701091019D+00,0.3848900399931313D+00,
     2  0.4010977213135840D+00,0.4223137033958720D+00,
     3  0.4494101241286116D+00,0.4851067166445962D+00,
     4  0.5338849940716039D+00,0.6038971937133730D+00,
     5  0.7154514704167375D+00,0.9537591536681720D+00/
c        
c        
c  Data for  12 nodes
c        
c        Nodes:
c        
      data tsm3/
     1  0.3623235566811749D+00,0.1089862350125027D+01,
     2  0.1826246105264088D+01,0.2577949657550823D+01,
     3  0.3352355815745010D+01,0.4158202198761708D+01,
     4  0.5006141390998915D+01,0.5909583144816792D+01,
     5  0.6886349769898109D+01,0.7962745685416317D+01,
     6  0.9185599279872974D+01,0.1067219949994418D+02/
c        
c        Weights:
c        
      data wsm3/
     1  0.7251262565254991D+00,0.7309377378900146D+00,
     2  0.7429025253831371D+00,0.7617276967542633D+00,
     3  0.7885318711687985D+00,0.8249233374404576D+00,
     4  0.8731655137385543D+00,0.9366301200668818D+00,
     5  0.1021125567183100D+01,0.1138914512058814D+01,
     6  0.1323221806465721D+01,0.1714899666972317D+01/
c        
c        
c  Data for  16 nodes
c        
c        Nodes:
c        
      data tsm2/
     1  0.5912891342026256D+00,0.1776441464280249D+01,
     2  0.2969390792362811D+01,0.4175584084775402D+01,
     3  0.5400839418882890D+01,0.6651487120878685D+01,
     4  0.7934500513835817D+01,0.9257618746009859D+01,
     5  0.1062948882016429D+02,0.1205990791309549D+02,
     6  0.1356034139488353D+02,0.1514505581679265D+02,
     7  0.1683357175040980D+02,0.1865625684925770D+02,
     8  0.2066934622616250D+02,0.2301290510694237D+02/
c        
c        Weights:
c        
      data wsm2/
     1  0.1183006031224341D+01,0.1188166627848316D+01,
     2  0.1198637395311120D+01,0.1214716503544217D+01,
     3  0.1236847038888007D+01,0.1265608591148965D+01,
     4  0.1301705863231786D+01,0.1345968125050024D+01,
     5  0.1399397001061102D+01,0.1463334662053937D+01,
     6  0.1539872670898909D+01,0.1632727115672674D+01,
     7  0.1749160236800245D+01,0.1904921709482490D+01,
     8  0.2141105727503994D+01,0.2621786502262962D+01/
c        
c        
c  Data for  25 nodes
c        
c        Nodes:
c        
      data tsm1/
     1  0.8362915183538084D+00,0.2510255191271542D+01,
     2  0.4188372130878084D+01,0.5873448670502245D+01,
     3  0.7568345988387720D+01,0.9275998970536598D+01,
     4  0.1099943138921065D+02,0.1274176572937194D+02,
     5  0.1450622605287869D+02,0.1629613302390456D+02,
     6  0.1811489200079322D+02,0.1996597811360477D+02,
     7  0.2185292623557653D+02,0.2377933794381220D+02,
     8  0.2574892101929257D+02,0.2776557950216954D+02,
     9  0.2983357514055825D+02,0.3195778788795812D+02,
     *  0.3414412078198387D+02,0.3640013651593878D+02,
     1  0.3873611110820030D+02,0.4116693515272880D+02,
     2  0.4371599640139223D+02,0.4642466359514905D+02,
     *  0.4938255488835708D+02/
c        
c        Weights:
c        
      data wsm1/
     1  0.1672812952594159D+01,0.1675576123993883D+01,
     2  0.1681125122548433D+01,0.1689504501995940D+01,
     3  0.1700779024377719D+01,0.1715030734090676D+01,
     4  0.1732354467166038D+01,0.1752851641341577D+01,
     5  0.1776622657070382D+01,0.1803759112144715D+01,
     6  0.1834338226255040D+01,0.1868423066435576D+01,
     7  0.1906072824949607D+01,0.1947367102346343D+01,
     8  0.1992447093114897D+01,0.2041575921819335D+01,
     9  0.2095222121843971D+01,0.2154176998941331D+01,
     *  0.2219732574264715D+01,0.2293982151661135D+01,
     1  0.2380390975475100D+01,0.2485024264373120D+01,
     2  0.2619645453153431D+01,0.2811839174484446D+01,
     *  0.3167807125537608D+01/
c        
c        
c  Data for  40 nodes
c        
c        Nodes:
c        
      data ts0/
     1  0.1057022601435076D+01,0.3174580060045362D+01,
     2  0.5301840942994462D+01,0.7443001390281078D+01,
     3  0.9600123634740653D+01,0.1177372053269191D+02,
     4  0.1396349501254189D+02,0.1616889671006253D+02,
     5  0.1838943463337256D+02,0.2062481005547635D+02,
     6  0.2287494753099967D+02,0.2513997822718924D+02,
     7  0.2742020560629174D+02,0.2971606784967717D+02,
     8  0.3202810305577201D+02,0.3435691932320878D+02,
     9  0.3670317022311467D+02,0.3906753563396112D+02,
     *  0.4145070781676176D+02,0.4385338264091316D+02,
     1  0.4627625590354923D+02,0.4872002468384849D+02,
     2  0.5118539365237398D+02,0.5367308624975114D+02,
     3  0.5618386070049412D+02,0.5871853097554341D+02,
     4  0.6127799310051405D+02,0.6386325767100604D+02,
     5  0.6647549015312808D+02,0.6911606164004935D+02,
     6  0.7178661447242972D+02,0.7448914990480002D+02,
     7  0.7722614993546064D+02,0.8000075419571277D+02,
     8  0.8281703054089877D+02,0.8568041502888467D+02,
     9  0.8859848596040754D+02,0.9158247466294083D+02,
     *  0.9465069531835854D+02,0.9784307745925156D+02/
c        
c        Weights:
c        
      data ws0/
     1  0.2114645941341827D+01,0.2121521570075472D+01,
     2  0.2133700831921736D+01,0.2148952450048807D+01,
     3  0.2165364400850742D+01,0.2181770537761840D+01,
     4  0.2197681680334000D+01,0.2213037864521845D+01,
     5  0.2227987804563856D+01,0.2242750810805510D+01,
     6  0.2257546604741697D+01,0.2272566003620492D+01,
     7  0.2287962369024749D+01,0.2303852300246646D+01,
     8  0.2320319870299242D+01,0.2337421912188235D+01,
     9  0.2355193432580969D+01,0.2373652912075041D+01,
     *  0.2392807492992541D+01,0.2412658098766676D+01,
     1  0.2433204501050298D+01,0.2454450318335065D+01,
     2  0.2476407925719686D+01,0.2499103291798145D+01,
     3  0.2522580836096182D+01,0.2546908515749295D+01,
     4  0.2572183507037088D+01,0.2598539056522236D+01,
     5  0.2626153381616836D+01,0.2655261975876254D+01,
     6  0.2686175507848209D+01,0.2719306954484497D+01,
     7  0.2755214524444577D+01,0.2794673325284067D+01,
     8  0.2838803746758098D+01,0.2889326993368821D+01,
     9  0.2949177744586291D+01,0.3024493845935104D+01,
     *  0.3135201393600549D+01,0.3462874979045710D+01/
c        

      if (ilev.ge.nlevels) goto 2000
      
      npw2=npw/2
      
      if (ilev.eq.0) then
         do i=1,npw2
            ts(i+npw2)=ts0(i)
            ws(i+npw2)=ws0(i)
         enddo
      elseif (ilev.eq.-1) then
         do i=1,npw2
            ts(i+npw2)=tsm1(i)
            ws(i+npw2)=wsm1(i)
         enddo
      elseif (ilev.eq.-2) then
         do i=1,npw2
            ts(i+npw2)=tsm2(i)
            ws(i+npw2)=wsm2(i)
         enddo
      elseif (ilev.eq.-3) then
         do i=1,npw2
            ts(i+npw2)=tsm3(i)
            ws(i+npw2)=wsm3(i)
         enddo
      elseif (ilev.eq.-4) then
         do i=1,npw2
            ts(i+npw2)=tsm4(i)
            ws(i+npw2)=wsm4(i)
         enddo
      elseif (ilev.eq.-5) then
         do i=1,npw2
            ts(i+npw2)=tsm5(i)
            ws(i+npw2)=wsm5(i)
         enddo
      elseif (ilev.eq.-6) then
         do i=1,npw2
            ts(i+npw2)=tsm6(i)
            ws(i+npw2)=wsm6(i)
         enddo
      elseif (ilev.eq.-7) then
         do i=1,npw2
            ts(i+npw2)=tsm7(i)
            ws(i+npw2)=wsm7(i)
         enddo
      elseif (ilev.eq.-8) then
         do i=1,npw2
            ts(i+npw2)=tsm8(i)
            ws(i+npw2)=wsm8(i)
         enddo
      elseif (ilev.eq.-9) then
         do i=1,npw2
            ts(i+npw2)=tsm9(i)
            ws(i+npw2)=wsm9(i)
         enddo
      elseif (ilev.eq.-10) then
         do i=1,npw2
            ts(i+npw2)=tsm10(i)
            ws(i+npw2)=wsm10(i)
         enddo
      elseif (ilev.gt.0) then
         do i=1,npw2
            ts(i+npw2)=ts0(i)*2**ilev
            ws(i+npw2)=ws0(i)*2**ilev
         enddo
      endif
      
      do i=1,npw2
         ts(npw2-i+1)=-ts(i+npw2)
         ws(npw2-i+1)=ws(i+npw2)
      enddo
c
      sqrtpi=sqrt(4*atan(1.0d0))
      do i=1,npw
         ws(i)=ws(i)/2/sqrtpi
      enddo

 2000 continue
         
      
      return
      end 
c
c
c
c
      
