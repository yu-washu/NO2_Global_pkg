        !COMPILER-GENERATED INTERFACE MODULE: Mon Jan 28 16:14:31 2019
        ! This source file is for reference only and may not completely
        ! represent the generated interface used by the compiler.
        MODULE SINGLE_TOPBOT_SLICER__genmod
          INTERFACE 
            SUBROUTINE SINGLE_TOPBOT_SLICER(YG,NYDIM,NYBOX,SIDE_A,PARS, &
     &DO_TOP,ORNT,JOLD_L,JOLD_U,JCNR,JNEW_L,JNEW_U,XOLD,XNEW,YOLD_L,    &
     &YOLD_U,YNEW_L,YNEW_U,XCNR,YCNR,AREA)
              INTEGER(KIND=4) :: NYBOX
              INTEGER(KIND=4) :: NYDIM
              REAL(KIND=8) :: YG(NYDIM)
              INTEGER(KIND=4) :: SIDE_A
              REAL(KIND=8) :: PARS(6,4)
              INTEGER(KIND=4) :: DO_TOP
              INTEGER(KIND=4) :: ORNT
              INTEGER(KIND=4) :: JOLD_L
              INTEGER(KIND=4) :: JOLD_U
              INTEGER(KIND=4) :: JCNR
              INTEGER(KIND=4) :: JNEW_L
              INTEGER(KIND=4) :: JNEW_U
              REAL(KIND=8) :: XOLD
              REAL(KIND=8) :: XNEW
              REAL(KIND=8) :: YOLD_L
              REAL(KIND=8) :: YOLD_U
              REAL(KIND=8) :: YNEW_L
              REAL(KIND=8) :: YNEW_U
              REAL(KIND=8) :: XCNR
              REAL(KIND=8) :: YCNR
              REAL(KIND=8) :: AREA(NYBOX)
            END SUBROUTINE SINGLE_TOPBOT_SLICER
          END INTERFACE 
        END MODULE SINGLE_TOPBOT_SLICER__genmod
