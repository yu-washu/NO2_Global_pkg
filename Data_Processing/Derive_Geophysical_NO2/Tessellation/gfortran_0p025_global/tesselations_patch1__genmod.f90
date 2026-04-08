        !COMPILER-GENERATED INTERFACE MODULE: Mon Jan 28 16:14:32 2019
        ! This source file is for reference only and may not completely
        ! represent the generated interface used by the compiler.
        MODULE TESSELATIONS_PATCH1__genmod
          INTERFACE 
            SUBROUTINE TESSELATIONS_PATCH1(MAXRUNS,MAX_YDIM,MAX_XDIM,   &
     &MAX_ADIM,NRUNS,FOOT_COORDS,GLOBAL_ALBEDO_1,ETOP05_CLIMDATA,OXGRID,&
     &OYGRID,FOOT_ALBEDOS,FOOT_HEIGHTS,FOOT_AREAS)
              INTEGER(KIND=4) :: MAX_ADIM
              INTEGER(KIND=4) :: MAX_XDIM
              INTEGER(KIND=4) :: MAX_YDIM
              INTEGER(KIND=4) :: MAXRUNS
              INTEGER(KIND=4) :: NRUNS
              REAL(KIND=8) :: FOOT_COORDS(5,2,MAXRUNS)
              REAL(KIND=4) :: GLOBAL_ALBEDO_1(MAX_YDIM,MAX_ADIM)
              INTEGER(KIND=4) :: ETOP05_CLIMDATA(MAX_YDIM,MAX_ADIM)
              REAL(KIND=8) :: OXGRID(MAX_XDIM)
              REAL(KIND=8) :: OYGRID(MAX_YDIM)
              REAL(KIND=8) :: FOOT_ALBEDOS(MAXRUNS)
              REAL(KIND=8) :: FOOT_HEIGHTS(MAXRUNS)
              REAL(KIND=8) :: FOOT_AREAS(MAXRUNS)
            END SUBROUTINE TESSELATIONS_PATCH1
          END INTERFACE 
        END MODULE TESSELATIONS_PATCH1__genmod
