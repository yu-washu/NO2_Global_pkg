        !COMPILER-GENERATED INTERFACE MODULE: Mon Jan 28 16:14:31 2019
        ! This source file is for reference only and may not completely
        ! represent the generated interface used by the compiler.
        MODULE DOUBLE_LEFT_SLICER__genmod
          INTERFACE 
            SUBROUTINE DOUBLE_LEFT_SLICER(YG,NYDIM,NYBOX,ORNT,PARS,     &
     &JNEW_L,JCNR_F,JCNR_D,JNEW_U,XNEW,YNEW_L,YNEW_U,XCNR_F,YCNR_F,     &
     &XCNR_D,YCNR_D,AREA)
              INTEGER(KIND=4) :: NYBOX
              INTEGER(KIND=4) :: NYDIM
              REAL(KIND=8) :: YG(NYDIM)
              INTEGER(KIND=4) :: ORNT
              REAL(KIND=8) :: PARS(6,4)
              INTEGER(KIND=4) :: JNEW_L
              INTEGER(KIND=4) :: JCNR_F
              INTEGER(KIND=4) :: JCNR_D
              INTEGER(KIND=4) :: JNEW_U
              REAL(KIND=8) :: XNEW
              REAL(KIND=8) :: YNEW_L
              REAL(KIND=8) :: YNEW_U
              REAL(KIND=8) :: XCNR_F
              REAL(KIND=8) :: YCNR_F
              REAL(KIND=8) :: XCNR_D
              REAL(KIND=8) :: YCNR_D
              REAL(KIND=8) :: AREA(NYBOX)
            END SUBROUTINE DOUBLE_LEFT_SLICER
          END INTERFACE 
        END MODULE DOUBLE_LEFT_SLICER__genmod
