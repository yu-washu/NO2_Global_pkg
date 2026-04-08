        !COMPILER-GENERATED INTERFACE MODULE: Mon Jan 28 16:14:31 2019
        ! This source file is for reference only and may not completely
        ! represent the generated interface used by the compiler.
        MODULE DOUBLESIDE_SLICER__genmod
          INTERFACE 
            SUBROUTINE DOUBLESIDE_SLICER(YG,NYDIM,NYBOX,SIDE_L,SIDE_U,  &
     &PARS,JOLD_L,JOLD_U,JNEW_L,JNEW_U,PARALLEL,ORNT,XOLD,XNEW,YOLD_L,  &
     &YOLD_U,YNEW_L,YNEW_U,AREA)
              INTEGER(KIND=4) :: NYBOX
              INTEGER(KIND=4) :: NYDIM
              REAL(KIND=8) :: YG(NYDIM)
              INTEGER(KIND=4) :: SIDE_L
              INTEGER(KIND=4) :: SIDE_U
              REAL(KIND=8) :: PARS(6,4)
              INTEGER(KIND=4) :: JOLD_L
              INTEGER(KIND=4) :: JOLD_U
              INTEGER(KIND=4) :: JNEW_L
              INTEGER(KIND=4) :: JNEW_U
              INTEGER(KIND=4) :: PARALLEL
              INTEGER(KIND=4) :: ORNT
              REAL(KIND=8) :: XOLD
              REAL(KIND=8) :: XNEW
              REAL(KIND=8) :: YOLD_L
              REAL(KIND=8) :: YOLD_U
              REAL(KIND=8) :: YNEW_L
              REAL(KIND=8) :: YNEW_U
              REAL(KIND=8) :: AREA(NYBOX)
            END SUBROUTINE DOUBLESIDE_SLICER
          END INTERFACE 
        END MODULE DOUBLESIDE_SLICER__genmod
