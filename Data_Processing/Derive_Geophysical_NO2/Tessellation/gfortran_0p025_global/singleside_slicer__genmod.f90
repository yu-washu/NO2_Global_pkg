        !COMPILER-GENERATED INTERFACE MODULE: Mon Jan 28 16:14:31 2019
        ! This source file is for reference only and may not completely
        ! represent the generated interface used by the compiler.
        MODULE SINGLESIDE_SLICER__genmod
          INTERFACE 
            SUBROUTINE SINGLESIDE_SLICER(YG,LOCAL_NYDIM,LOCAL_NYBOX,SIDE&
     &,PARS,JOLD,JNEW,XOLD,YOLD,XNEW,YNEW,AREA)
              INTEGER(KIND=4) :: LOCAL_NYBOX
              INTEGER(KIND=4) :: LOCAL_NYDIM
              REAL(KIND=8) :: YG(LOCAL_NYDIM)
              INTEGER(KIND=4) :: SIDE
              REAL(KIND=8) :: PARS(6,4)
              INTEGER(KIND=4) :: JOLD
              INTEGER(KIND=4) :: JNEW
              REAL(KIND=8) :: XOLD
              REAL(KIND=8) :: YOLD
              REAL(KIND=8) :: XNEW
              REAL(KIND=8) :: YNEW
              REAL(KIND=8) :: AREA(LOCAL_NYBOX)
            END SUBROUTINE SINGLESIDE_SLICER
          END INTERFACE 
        END MODULE SINGLESIDE_SLICER__genmod
