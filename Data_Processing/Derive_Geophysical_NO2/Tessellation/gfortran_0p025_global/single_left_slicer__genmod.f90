        !COMPILER-GENERATED INTERFACE MODULE: Mon Jan 28 16:14:31 2019
        ! This source file is for reference only and may not completely
        ! represent the generated interface used by the compiler.
        MODULE SINGLE_LEFT_SLICER__genmod
          INTERFACE 
            SUBROUTINE SINGLE_LEFT_SLICER(YG,NYDIM,NYBOX,PARS,JNEW_L,   &
     &JCNR,JNEW_U,XNEW,YNEW_L,YNEW_U,XCNR,YCNR,AREA)
              INTEGER(KIND=4) :: NYBOX
              INTEGER(KIND=4) :: NYDIM
              REAL(KIND=8) :: YG(NYDIM)
              REAL(KIND=8) :: PARS(6,4)
              INTEGER(KIND=4) :: JNEW_L
              INTEGER(KIND=4) :: JCNR
              INTEGER(KIND=4) :: JNEW_U
              REAL(KIND=8) :: XNEW
              REAL(KIND=8) :: YNEW_L
              REAL(KIND=8) :: YNEW_U
              REAL(KIND=8) :: XCNR
              REAL(KIND=8) :: YCNR
              REAL(KIND=8) :: AREA(NYBOX)
            END SUBROUTINE SINGLE_LEFT_SLICER
          END INTERFACE 
        END MODULE SINGLE_LEFT_SLICER__genmod
