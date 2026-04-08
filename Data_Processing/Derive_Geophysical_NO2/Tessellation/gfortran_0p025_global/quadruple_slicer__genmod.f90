        !COMPILER-GENERATED INTERFACE MODULE: Mon Jan 28 16:14:31 2019
        ! This source file is for reference only and may not completely
        ! represent the generated interface used by the compiler.
        MODULE QUADRUPLE_SLICER__genmod
          INTERFACE 
            SUBROUTINE QUADRUPLE_SLICER(YG,NYDIM,NYBOX,ORNT,PARS,JC1,JC2&
     &,JC3,JC4,XC1,YC1,XC2,YC2,XC3,YC3,XC4,YC4,AREA)
              INTEGER(KIND=4) :: NYBOX
              INTEGER(KIND=4) :: NYDIM
              REAL(KIND=8) :: YG(NYDIM)
              INTEGER(KIND=4) :: ORNT
              REAL(KIND=8) :: PARS(6,4)
              INTEGER(KIND=4) :: JC1
              INTEGER(KIND=4) :: JC2
              INTEGER(KIND=4) :: JC3
              INTEGER(KIND=4) :: JC4
              REAL(KIND=8) :: XC1
              REAL(KIND=8) :: YC1
              REAL(KIND=8) :: XC2
              REAL(KIND=8) :: YC2
              REAL(KIND=8) :: XC3
              REAL(KIND=8) :: YC3
              REAL(KIND=8) :: XC4
              REAL(KIND=8) :: YC4
              REAL(KIND=8) :: AREA(NYBOX)
            END SUBROUTINE QUADRUPLE_SLICER
          END INTERFACE 
        END MODULE QUADRUPLE_SLICER__genmod
