#ifndef _COULOMB_
#define _COULOMB_

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#define CUDA_CALLABLE_MEMBER_DEVICE __device__
#else
#define CUDA_CALLABLE_MEMBER
#define CUDA_CALLABLE_MEMBER_DEVICE
#endif

#include "Particles.h"
#include <cmath>
#ifdef __CUDACC__
#include <thrust/random.h>
#else
#include <random>
#endif

CUDA_CALLABLE_MEMBER
void getSlowDownFrequencies ( float& nu_friction, float& nu_deflection, float& nu_parallel,
			 	float& nu_energy, float x, float y,float z, float vx, float vy, float vz,float charge, float amu, 
                
    int nR_flowV,
    int nZ_flowV,
    float* flowVGridr,
    float* flowVGridz,
    float* flowVr,
    float* flowVz,
    float* flowVt,
    int nR_Dens,
    int nZ_Dens,
    float* DensGridr,
    float* DensGridz,
    float* ni,
    int nR_Temp,
    int nZ_Temp,
    float* TempGridr,
    float* TempGridz,
    float* ti, float* te,float background_Z, float background_amu,
                        int nR_Bfield, int nZ_Bfield,
                        float* BfieldGridR ,float* BfieldGridZ ,
                        float* BfieldR ,float* BfieldZ ,
                 float* BfieldT,float &T_background 
                ) {
        float Q = 1.60217662e-19;
        float EPS0 = 8.854187e-12;
	float pi = 3.14159265;
float MI = 1.6737236e-27;	
float ME = 9.10938356e-31;
        float te_eV = interp2dCombined(x,y,z,nR_Temp,nZ_Temp,TempGridr,TempGridz,te);
        float ti_eV = interp2dCombined(x,y,z,nR_Temp,nZ_Temp,TempGridr,TempGridz,ti);
	T_background = ti_eV;
            float density = interp2dCombined(x,y,z,nR_Dens,nZ_Dens,DensGridr,DensGridz,ni);
            //std::cout << "ion t and n " << te_eV << "  " << density << std::endl;
    float flowVelocity[3]= {0.0f};
	float relativeVelocity[3] = {0.0, 0.0, 0.0};
	float velocityNorm = 0.0f;
	float lam_d;
	float lam;
	float gam_electron_background;
	float gam_ion_background;
	float a_electron = 0.0;
	float a_ion = 0.0;
	float xx;
	float psi_prime;
	float psi_psiprime;
	float psi;
	float xx_e;
	float psi_prime_e;
	float psi_psiprime_e;
	float psi_psiprime_psi2x = 0.0;
	float psi_psiprime_psi2x_e = 0.0;
	float psi_e;
	float nu_0_i;
	float nu_0_e;
    float nu_friction_i; 
    float nu_deflection_i;
    float nu_parallel_i; 
    float nu_energy_i;
    float nu_friction_e; 
    float nu_deflection_e;
    float nu_parallel_e; 
    float nu_energy_e;
                
#if FLOWV_INTERP == 3
        interp3dVector (&flowVelocity[0], x,y,z,nR_flowV,nY_flowV,nZ_flowV,
                flowVGridr,flowVGridy,flowVGridz,flowVr,flowVz,flowVt);
#elif FLOWV_INTERP < 3    
#if USEFIELDALIGNEDVALUES > 0
        interpFieldAlignedVector(&flowVelocity[0],x,y,z,
                                 nR_flowV,nZ_flowV,
                                 flowVGridr,flowVGridz,flowVr,
                                 flowVz,flowVt,nR_Bfield,nZ_Bfield,
                                 BfieldGridR,BfieldGridZ,BfieldR,
                                 BfieldZ,BfieldT);
#else
               interp2dVector(&flowVelocity[0],x,y,z,
                        nR_flowV,nZ_flowV,
                        flowVGridr,flowVGridz,flowVr,flowVz,flowVt);
#endif
#endif
	relativeVelocity[0] = vx - flowVelocity[0];
	relativeVelocity[1] = vy - flowVelocity[1];
	relativeVelocity[2] = vz - flowVelocity[2];
	velocityNorm = std::sqrt( relativeVelocity[0]*relativeVelocity[0] + relativeVelocity[1]*relativeVelocity[1] + relativeVelocity[2]*relativeVelocity[2]);                
	    //std::cout << "velocity norm " << velocityNorm << std::endl;	
    //for(int i=1; i < nSpecies; i++)
		//{
			lam_d = std::sqrt(EPS0*te_eV/(density*std::pow(background_Z,2)*Q));//only one q in order to convert to J
                	lam = 4.0*pi*density*std::pow(lam_d,3);
                	gam_electron_background = 0.238762895*std::pow(charge,2)*std::log(lam)/(amu*amu);//constant = Q^4/(MI^2*4*pi*EPS0^2)
                	gam_ion_background = 0.238762895*std::pow(charge,2)*std::pow(background_Z,2)*std::log(lam)/(amu*amu);//constant = Q^4/(MI^2*4*pi*EPS0^2)
                    //std::cout << "gam components " <<gam_electron_background << " " << pow(Q,4) << " " << " " << pow(background_Z,2) << " " << log(lam)<< std::endl; 
                if(gam_electron_background < 0.0) gam_electron_background=0.0;
                if(gam_ion_background < 0.0) gam_ion_background=0.0;
		       a_ion = background_amu*MI/(2*ti_eV*Q);// %q is just to convert units - no z needed
                	a_electron = ME/(2*te_eV*Q);// %q is just to convert units - no z needed
                
                	xx = std::pow(velocityNorm,2)*a_ion;
                	//psi_prime = 2*sqrtf(xx/pi)*expf(-xx);
                	//psi_psiprime = erf(1.0*sqrtf(xx));
                	//psi = psi_psiprime - psi_prime;
                        //if(psi < 0.0) psi = 0.0;
		        //psi_psiprime_psi2x = psi+psi_prime - psi/2.0/x;
		        //if(xx<1.0e-3)
		        //{
                            psi_prime = 1.128379*std::sqrt(xx);
                            psi = 0.75225278*std::pow(xx,1.5);
                            psi_psiprime = psi+psi_prime;
                            psi_psiprime_psi2x = 1.128379*std::sqrt(xx)*expf(-xx);
		        //}
                    //if(psi_prime/psi > 1.0e7) psi = psi_psiprime/1.0e7;
                    //if(psi_prime < 0.0) psi_prime = 0.0;
                    //if(psi_psiprime < 0.0) psi_psiprime = 0.0;
                	xx_e = std::pow(velocityNorm,2)*a_electron;
                	//psi_prime_e = 2*sqrtf(xx_e/pi)*expf(-xx_e);
                	//psi_psiprime_e = erf(1.0*sqrtf(xx_e));
                	//psi_e = psi_psiprime_e - psi_prime_e;
                        //if(psi_e < 0.0) psi_e = 0.0;
		        //psi_psiprime_psi2x_e = psi_e+psi_prime_e - psi_e/2.0/xx_e;
		        //if(xx_e<1.0e-3)
		        //{
                            psi_prime_e = 1.128379*std::sqrt(xx_e);
                            psi_e = 0.75225278*std::pow(xx_e,1.5);
                            psi_psiprime_e = psi_e+psi_prime_e;
                            psi_psiprime_psi2x_e = 1.128379*std::sqrt(xx_e)*expf(-xx_e);
		        //}
                    //if(psi_prime_e/psi_e > 1.0e7) psi_e = psi_psiprime_e/1.0e7;
                    //if(psi_prime_e < 0.0) psi_prime_e = 0.0;
                    //if(psi_psiprime_e < 0.0) psi_psiprime_e = 0.0;
                	nu_0_i = gam_electron_background*density/std::pow(velocityNorm,3);
                	nu_0_e = gam_ion_background*density/std::pow(velocityNorm,3);
                	nu_friction_i = (1+amu/background_amu)*psi*nu_0_i;
                	//nu_deflection_i = 2*(psi_psiprime - psi/(2*xx))*nu_0_i;
                	nu_deflection_i = 2*(psi_psiprime_psi2x)*nu_0_i;
                	nu_parallel_i = psi/xx*nu_0_i;
                	nu_energy_i = 2*(amu/background_amu*psi - psi_prime)*nu_0_i;
                	nu_friction_e = (1+amu/(ME/MI))*psi_e*nu_0_e;
                	//nu_deflection_e = 2*(psi_psiprime_e - psi_e/(2*xx_e))*nu_0_e;
                	nu_deflection_e = 2*(psi_psiprime_psi2x_e)*nu_0_e;
                	nu_parallel_e = psi_e/xx_e*nu_0_e;
                	nu_energy_e = 2*(amu/(ME/MI)*psi_e - psi_prime_e)*nu_0_e;
                    
       //if(isnan(nu_friction_i)){
       //std::cout << "nu_f_i " << nu_friction_i << std::endl;
       //std::cout << " psi " << psi;
       //std::cout << " xx " << xx;
       //std::cout << " ti_eV " << ti_eV;
       //std::cout << " a_ion " << a_ion << std::endl;
       //std::cout << "nu_0_i " << nu_0_i << std::endl;
       //}
       //if(isnan(nu_friction_e)) std::cout << "nu_f_e " << nu_friction_e << std::endl;
		    //std::cout << "lam_d lam gami game ai ae" << lam_d << " " << lam << " " << gam_ion_background << " " << gam_electron_background << " " << a_ion << " " << a_electron << std::endl;
                    //std::cout << "x psi_prime psi_psiprime psi" << xx << " " << xx_e << " " << psi_prime << " "<< psi_prime_e << " " << psi_psiprime<< " " << psi_psiprime_e << " " << psi<< " " << psi_e << " " << nu_0_i<< " " << nu_0_e << std::endl;
                    //std::cout << "nu friction, parallel perp energy IONs" << nu_friction_i << " " << nu_parallel_i << " " <<nu_deflection_i << " " << nu_energy_i << std::endl;
                    //std::cout << "nu friction, parallel perp energy ELECTRONs" << nu_friction_e << " " << nu_parallel_e << " " <<nu_deflection_e << " " << nu_energy_e << std::endl;
	//	}
    nu_friction = nu_friction_i + nu_friction_e;
    nu_deflection = nu_deflection_i + nu_deflection_e;
    nu_parallel = nu_parallel_i + nu_parallel_e;
    nu_energy = nu_energy_i + nu_energy_e;
     //if(nu_deflection < 0.0){
     //                std::cout << "nu0 "  << " " <<nu_0_i << " " << nu_0_e << " " << psi_psiprime_psi2x << " " << psi_psiprime_psi2x_e << std::endl;
     //    	    std::cout << "gam_electron_background*density/powf(velocityNorm,3) " << gam_electron_background<< " " << density<< " " << velocityNorm << " " << lam_d<< std::endl;
     //    	    }
     //                //std::cout << "nu friction, parallel perp energy " << nu_friction << " " << nu_parallel << " " <<nu_deflection << " " << nu_energy << std::endl;
    if(te_eV <= 0.0 || ti_eV <= 0.0)
    {
        nu_friction = 0.0;
        nu_deflection = 0.0;
        nu_parallel = 0.0;
        nu_energy = 0.0;
       //std::cout << " ti_eV and nu_friction " << ti_eV<< " " << nu_friction << std::endl;
    }
    if(density <= 0.0)
    {
        nu_friction = 0.0;
        nu_deflection = 0.0;
        nu_parallel = 0.0;
        nu_energy = 0.0;
    }

    //if(abs(nu_energy) > 1.0e7) 
    //{
    //        std::cout << "velocity norm " << velocityNorm << std::endl;	
    //std::cout << "gams " << gam_electron_background << " " << gam_ion_background << std::endl;
    //std::cout << " a and xx " << a_electron <<" " <<  a_ion<< " " << xx << " " << xx_e << std::endl;
    //       std::cout << "psi_prime psi_psiprime psi" << psi_prime << " "<< psi_prime_e << " " << psi_psiprime<< " " << psi_psiprime_e << " " << psi<< " " << psi_e << std::endl;
    //std::cout << "nu_Ei and nuEe " << nu_energy_i << " " << nu_energy_e << std::endl;
    // std::cout << "nu0 "  << " " <<nu_0_i << " " << nu_0_e << " " << psi_psiprime_psi2x << " " << psi_psiprime_psi2x_e << std::endl;
    ////nu_energy = vx;
    ////nu_friction = vy;
    ////nu_deflection = vz;
    ////std::cout << "nu friction i e total " << nu_deflection_i << " " << nu_deflection_e << " " <<nu_deflection  << std::endl;
    //}
}

CUDA_CALLABLE_MEMBER
void getSlowDownDirections (float parallel_direction[], float perp_direction1[], float perp_direction2[],
        float xprevious, float yprevious, float zprevious,float vx, float vy, float vz,
    int nR_flowV,
    int nY_flowV,
    int nZ_flowV,
    float* flowVGridr,
    float* flowVGridy,
    float* flowVGridz,
    float* flowVr,
    float* flowVz,
    float* flowVt,
    
                        int nR_Bfield, int nZ_Bfield,
                        float* BfieldGridR ,float* BfieldGridZ ,
                        float* BfieldR ,float* BfieldZ ,
                 float* BfieldT 
    ) {
	        float flowVelocity[3]= {0.0f};
                float relativeVelocity[3] = {0.0, 0.0, 0.0};
                float B[3] = {0.0f};
                float Bmag = 0.0;
		float B_unit[3] = {0.0};
		float velocityRelativeNorm;
		float s1;
		float s2;
        interp2dVector(&B[0],xprevious,yprevious,zprevious,nR_Bfield,nZ_Bfield,
                                       BfieldGridR,BfieldGridZ,BfieldR,BfieldZ,BfieldT);
        Bmag = std::sqrt(B[0]*B[0] + B[1]*B[1] + B[2]*B[2]);
        B_unit[0] = B[0]/Bmag;
        B_unit[1] = B[1]/Bmag;
        B_unit[2] = B[2]/Bmag;
        if(Bmag ==0.0)
        {Bmag = 1.0;
            B_unit[0] = 1.0;
            B_unit[1] = 0.0;
            B_unit[2] = 0.0;
        }

#if FLOWV_INTERP == 3
        interp3dVector (&flowVelocity[0], xprevious,yprevious,zprevious,nR_flowV,nY_flowV,nZ_flowV,
                flowVGridr,flowVGridy,flowVGridz,flowVr,flowVz,flowVt);
#elif FLOWV_INTERP < 3    
#if USEFIELDALIGNEDVALUES > 0
        interpFieldAlignedVector(flowVelocity,xprevious,yprevious,
                                 zprevious,nR_flowV,nZ_flowV,
                                 flowVGridr,flowVGridz,flowVr,
                                 flowVz,flowVt,nR_Bfield,nZ_Bfield,
                                 BfieldGridR,BfieldGridZ,BfieldR,
                                 BfieldZ,BfieldT);
#else
               interp2dVector(&flowVelocity[0],xprevious,yprevious,zprevious,
                        nR_flowV,nZ_flowV,
                        flowVGridr,flowVGridz,flowVr,flowVz,flowVt);
#endif
#endif
                relativeVelocity[0] = vx;// - flowVelocity[0];
                relativeVelocity[1] = vy;// - flowVelocity[1];
                relativeVelocity[2] = vz;// - flowVelocity[2];
                velocityRelativeNorm = std::sqrt( relativeVelocity[0]*relativeVelocity[0] + relativeVelocity[1]*relativeVelocity[1] + relativeVelocity[2]*relativeVelocity[2]);

		parallel_direction[0] = relativeVelocity[0]/velocityRelativeNorm;
		parallel_direction[1] = relativeVelocity[1]/velocityRelativeNorm;
		parallel_direction[2] = relativeVelocity[2]/velocityRelativeNorm;

		s1 = parallel_direction[0]*B_unit[0]+parallel_direction[1]*B_unit[1]+parallel_direction[2]*B_unit[2];
            	s2 = std::sqrt(1.0-s1*s1);
		//std::cout << "s1 and s2 " << s1 << " " << s2 << std::endl;
           if(std::abs(s1) >=1.0) s2=0; 
            	perp_direction1[0] = 1.0/s2*(s1*parallel_direction[0] - B_unit[0]);
		perp_direction1[1] = 1.0/s2*(s1*parallel_direction[1] - B_unit[1]);
		perp_direction1[2] = 1.0/s2*(s1*parallel_direction[2] - B_unit[2]);
           
                perp_direction2[0] = 1.0/s2*(parallel_direction[1]*B_unit[2] - parallel_direction[2]*B_unit[1]);
                perp_direction2[1] = 1.0/s2*(parallel_direction[2]*B_unit[0] - parallel_direction[0]*B_unit[2]);
                perp_direction2[2] = 1.0/s2*(parallel_direction[0]*B_unit[1] - parallel_direction[1]*B_unit[0]);
        //std::cout << "SlowdonwDir par" << parallel_direction[0] << " " << parallel_direction[1] << " " << parallel_direction[2] << " " << std::endl;
        //std::cout << "SlowdonwDir perp" << perp_direction1[0] << " " <<perp_direction1[1] << " " << perp_direction1[2] << " " << std::endl;
        //std::cout << "SlowdonwDir perp" << perp_direction2[0] << " " << perp_direction2[1] << " " << perp_direction2[2] << " " << std::endl;
            //perp_direction1[0] =  s1;
            //perp_direction1[1] =  s2;
            if (s2 == 0.0)
            {
                perp_direction1[0] =  s1;
                perp_direction1[1] =  s2;

                perp_direction2[0] = parallel_direction[2];
		perp_direction2[1] = parallel_direction[0];
		perp_direction2[2] = parallel_direction[1];

                s1 = parallel_direction[0]*perp_direction2[0]+parallel_direction[1]*perp_direction2[1]+parallel_direction[2]*perp_direction2[2];
                s2 = std::sqrt(1.0-s1*s1);
		perp_direction1[0] = -1.0/s2*(parallel_direction[1]*perp_direction2[2] - parallel_direction[2]*perp_direction2[1]);
                perp_direction1[1] = -1.0/s2*(parallel_direction[2]*perp_direction2[0] - parallel_direction[0]*perp_direction2[2]);
                perp_direction1[2] = -1.0/s2*(parallel_direction[0]*perp_direction2[1] - parallel_direction[1]*perp_direction2[0]);
            }
	    //if(parallel_direction[0]*parallel_direction[0] + parallel_direction[1]*parallel_direction[1] + parallel_direction[2]*parallel_direction[2] - 1.0 > 1e-6) std::cout << " parallel direction not one " << parallel_direction[0] << " " << parallel_direction[1] << " " << parallel_direction[2] << std::endl;
	    //if(perp_direction1[0]*perp_direction1[0] + perp_direction1[1]*perp_direction1[1] + perp_direction1[2]*perp_direction1[2] - 1.0 > 1e-6) std::cout << " perp direction1 not one" << perp_direction1[0] << " " << perp_direction1[1] << " " << perp_direction1[2] << std::endl;
	    //if(perp_direction2[0]*perp_direction2[0] + perp_direction2[1]*perp_direction2[1] + perp_direction2[2]*perp_direction2[2] - 1.0 > 1e-6) std::cout << " perp direction2 not one" << perp_direction2[0] << " " << perp_direction2[1] << " " << perp_direction2[2] << std::endl;
	    //if(vectorDotProduct(parallel_direction,perp_direction1)  > 1e-6)
	    //{std::cout << "par dot perp1 " << parallel_direction[0] << " " << parallel_direction[1] << " " << parallel_direction[2] << std::endl;
	    //std::cout << "par dot perp1 " << perp_direction1[0] << " " << perp_direction1[1] << " " << perp_direction1[2] << std::endl;
	    //}
	    //if(vectorDotProduct(parallel_direction,perp_direction2)  > 2e-6)
	    //{std::cout << "par dot perp2 " << parallel_direction[0] << " " << parallel_direction[2] << " " << parallel_direction[2] << std::endl;
	    //std::cout << "par dot perp2 " << perp_direction2[0] << " " << perp_direction2[2] << " " << perp_direction2[2] << std::endl;
	    //}
	    //if(vectorDotProduct(perp_direction1,perp_direction2)  > 2e-6)
	    //{std::cout << "perp1 dot perp2 " << perp_direction1[0] << " " << perp_direction1[2] << " " << perp_direction1[2] << std::endl;
	    //std::cout << "par dot perp2 " << perp_direction2[0] << " " << perp_direction2[2] << " " << perp_direction2[2] << std::endl;
	    //}
}

struct coulombCollisions { 
    Particles *particlesPointer;
    const float dt;
    int nR_flowV;
    int nY_flowV;
    int nZ_flowV;
    float* flowVGridr;
    float* flowVGridy;
    float* flowVGridz;
    float* flowVr;
    float* flowVz;
    float* flowVt;
    int nR_Dens;
    int nZ_Dens;
    float* DensGridr;
    float* DensGridz;
    float* ni;
    int nR_Temp;
    int nZ_Temp;
    float* TempGridr;
    float* TempGridz;
    float* ti;
    float* te;
    float background_Z;
    float background_amu;
    int nR_Bfield;
    int nZ_Bfield;
    float * BfieldGridR;
    float * BfieldGridZ;
    float * BfieldR;
    float * BfieldZ;
    float * BfieldT;
    float dv[3];
#if __CUDACC__
            curandState *state;
#else
            std::mt19937 *state;
#endif

    coulombCollisions(Particles *_particlesPointer,float _dt, 
#if __CUDACC__
                            curandState *_state,
#else
                            std::mt19937 *_state,
#endif
            int _nR_flowV,int _nY_flowV, int _nZ_flowV,    float* _flowVGridr,float* _flowVGridy,
                float* _flowVGridz,float* _flowVr,
                        float* _flowVz,float* _flowVt,
                        int _nR_Dens,int _nZ_Dens,float* _DensGridr,
                            float* _DensGridz,float* _ni,int _nR_Temp, int _nZ_Temp,
                        float* _TempGridr, float* _TempGridz,float* _ti,float* _te,
                        float _background_Z, float _background_amu,
                        int _nR_Bfield, int _nZ_Bfield,
                        float * _BfieldGridR ,float * _BfieldGridZ ,
                        float * _BfieldR ,float * _BfieldZ ,
                 float * _BfieldT )
      : particlesPointer(_particlesPointer),
        dt(_dt),
        nR_flowV(_nR_flowV),
        nY_flowV(_nY_flowV),
        nZ_flowV(_nZ_flowV),
        flowVGridr(_flowVGridr),
        flowVGridy(_flowVGridy),
        flowVGridz(_flowVGridz),
        flowVr(_flowVr),
        flowVz(_flowVz),
        flowVt(_flowVt),
        nR_Dens(_nR_Dens),
        nZ_Dens(_nZ_Dens),
        DensGridr(_DensGridr),
        DensGridz(_DensGridz),
        ni(_ni),
        nR_Temp(_nR_Temp),
        nZ_Temp(_nZ_Temp),
        TempGridr(_TempGridr),
        TempGridz(_TempGridz),
        ti(_ti),
        te(_te),
        background_Z(_background_Z),
        background_amu(_background_amu),
        nR_Bfield(_nR_Bfield),
        nZ_Bfield(_nZ_Bfield),
        BfieldGridR(_BfieldGridR),
        BfieldGridZ(_BfieldGridZ),
        BfieldR(_BfieldR),
        BfieldZ(_BfieldZ),
        BfieldT(_BfieldT),
        dv{0.0f, 0.0f, 0.0f},
        state(_state) {
  }
CUDA_CALLABLE_MEMBER_DEVICE    
void operator()(std::size_t indx)  { 

	    if(particlesPointer->hitWall[indx] == 0.0 && particlesPointer->charge[indx] != 0.0)
        { 
        float pi = 3.14159265;   
	float k_boltz = 1.38e-23*11604/1.66e-27;
	float T_background = 0.0;
		float nu_friction = 0.0;
		float nu_deflection = 0.0;
		float nu_parallel = 0.0;
		float nu_energy = 0.0;
		float flowVelocity[3]= {0.0f};
		float vUpdate[3]= {0.0f};
		float relativeVelocity[3] = {0.0f};
		float velocityCollisions[3]= {0.0f};	
		float velocityRelativeNorm;	
		float parallel_direction[3] = {0.0f};
		float parallel_direction_lab[3] = {0.0f};
		float perp_direction1[3] = {0.0f};
		float perp_direction2[3] = {0.0f};
		float parallel_contribution;
		float dv_perp1[3] = {0.0f};
		float dv_perp2[3] = {0.0f};
        float x = particlesPointer->xprevious[indx];
        float y = particlesPointer->yprevious[indx];
        float z = particlesPointer->zprevious[indx];
        float vx = particlesPointer->vx[indx];
        float vy = particlesPointer->vy[indx];
        float vz = particlesPointer->vz[indx];
        float vPartNorm = 0.0f;
#if FLOWV_INTERP == 3 
        interp3dVector (&flowVelocity[0], particlesPointer->xprevious[indx],particlesPointer->yprevious[indx],particlesPointer->zprevious[indx],nR_flowV,nY_flowV,nZ_flowV,
                flowVGridr,flowVGridy,flowVGridz,flowVr,flowVz,flowVt);
#elif FLOWV_INTERP < 3    
#if USEFIELDALIGNEDVALUES > 0
        interpFieldAlignedVector(&flowVelocity[0],
                                 particlesPointer->xprevious[indx],particlesPointer->yprevious[indx],particlesPointer->zprevious[indx],
                                 nR_flowV,nZ_flowV,
                                 flowVGridr,flowVGridz,flowVr,
                                 flowVz,flowVt,nR_Bfield,nZ_Bfield,
                                 BfieldGridR,BfieldGridZ,BfieldR,
                                 BfieldZ,BfieldT);
#else
               interp2dVector(flowVelocity,particlesPointer->xprevious[indx],particlesPointer->yprevious[indx],particlesPointer->zprevious[indx],
                        nR_flowV,nZ_flowV,
                        flowVGridr,flowVGridz,flowVr,flowVz,flowVt);
#endif
#endif
            vPartNorm = std::sqrt(vx*vx + vy*vy + vz*vz);

           //SFT 
	        relativeVelocity[0] = vx - flowVelocity[0];
        	relativeVelocity[1] = vy - flowVelocity[1];
        	relativeVelocity[2] = vz - flowVelocity[2];
		float vRel2 = relativeVelocity[0]*relativeVelocity[0] + relativeVelocity[1]*relativeVelocity[1] + relativeVelocity[2]*relativeVelocity[2];
        	velocityRelativeNorm = vectorNorm(relativeVelocity);
            //std::cout << "particle velocity " << particlesPointer->vx[indx] << " " << particlesPointer->vy[indx] << " " << particlesPointer->vz[indx] << std::endl;
            //std::cout << "flow velocity " << flowVelocity[0] << " " << flowVelocity[1] << " " << flowVelocity[2] << std::endl;
            //std::cout << "relative velocity and norm " << relativeVelocity[0] << " " << relativeVelocity[1] << " " << relativeVelocity[2]<< " " << velocityRelativeNorm << std::endl;
#if PARTICLESEEDS > 0
#ifdef __CUDACC__
        //int plus_minus1 = 1;//floor(curand_uniform(&particlesPointer->streams_collision1[indx]) + 0.5)*2 -1;
		//int plus_minus2 = 1;//floor(curand_uniform(&particlesPointer->streams_collision2[indx]) + 0.5)*2 -1;
		//int plus_minus3 = 1;//floor(curand_uniform(&particlesPointer->streams_collision3[indx]) + 0.5)*2 -1;
            float n1 = curand_normal(&state[indx]);
            float n2 = curand_normal(&state[indx]);
            float r1 = curand_uniform(&state[indx]);
            float r2 = curand_uniform(&state[indx]);
            float r3 = curand_uniform(&state[indx]);
            float xsi = curand_uniform(&state[indx]);
            //particlesPointer->test[indx] = xsi;
#else
	    //std::uniform_real_distribution<float> dist(0.0, 1.0);
        //int plus_minus1 = floor(dist(state[indx]) + 0.5)*2 - 1;
		//int plus_minus2 = floor(dist(state[indx]) + 0.5)*2 - 1;
		//int plus_minus3 = floor(dist(state[indx]) + 0.5)*2 - 1;
            std::normal_distribution<double> distribution(0.0,1.0);
            std::uniform_real_distribution<float> dist(0.0, 1.0);
            float n1 = distribution(state[indx]);
            float n2 = distribution(state[indx]);
            float r1 = dist(state[indx]);
            float r2 = dist(state[indx]);
            float r3 = dist(state[indx]);
            float xsi = dist(state[indx]);
#endif
#else
#if __CUDACC__
            //float plus_minus1 = floor(curand_uniform(&state[3]) + 0.5)*2-1;
            //float plus_minus2 = floor(curand_uniform(&state[4]) + 0.5)*2-1;
            //float plus_minus3 = floor(curand_uniform(&state[5]) + 0.5)*2-1;
            float n2 = curand_normal(&state[indx]);
            float xsi = curand_uniform(&state[indx]);
#else
            std::normal_distribution<double> distribution(0.0,1.0);
            std::uniform_real_distribution<float> dist(0.0, 1.0);
            float n2 = distribution(state[indx]);
            float xsi = dist(state[indx]);
            //float plus_minus1 = floor(dist(state[3]) + 0.5)*2 - 1;
            //float plus_minus2 = floor(dist(state[4]) + 0.5)*2 - 1;
            //float plus_minus3 = floor(dist(state[5]) + 0.5)*2 - 1;
#endif
#endif
      //std::cout << "random numbers n2 and xsi " << n2 << " " << xsi  << std::endl;

      //std::cout << "particle v " << particlesPointer->vx[indx] << " " << particlesPointer->vy[indx] << " " << particlesPointer->vz[indx] << std::endl;
      //        std::cout << "speed " << velocityRelativeNorm << std::endl;

      getSlowDownFrequencies(nu_friction, nu_deflection, nu_parallel, nu_energy,
                             x, y, z,
                             vx, vy, vz,
                             particlesPointer->charge[indx], particlesPointer->amu[indx],
                             nR_flowV, nZ_flowV, flowVGridr,
                             flowVGridz, flowVr,
                             flowVz, flowVt,
                             nR_Dens, nZ_Dens, DensGridr,
                             DensGridz, ni, nR_Temp, nZ_Temp,
                             TempGridr, TempGridz, ti, te, background_Z, background_amu,
                             nR_Bfield,
                             nZ_Bfield,
                             BfieldGridR,
                             BfieldGridZ,
                             BfieldR,
                             BfieldZ,
                             BfieldT, T_background);

      getSlowDownDirections(parallel_direction, perp_direction1, perp_direction2,
                            x, y, z,
                            vx, vy, vz,
                            nR_flowV, nY_flowV, nZ_flowV, flowVGridr, flowVGridy,
                            flowVGridz, flowVr,
                            flowVz, flowVt,

                            nR_Bfield,
                            nZ_Bfield,
                            BfieldGridR,
                            BfieldGridZ,
                            BfieldR,
                            BfieldZ,
                            BfieldT);

      //std::cout << "Velocity z " << particlesPointer->vz[indx] << endl;
      //std::cout << "SlowdonwDir par" << parallel_direction[0] << " " << parallel_direction[1] << " " << parallel_direction[2] << " " << std::endl;
      //std::cout << "SlowdonwDir perp" << perp_direction1[0] << " " <<perp_direction1[1] << " " << perp_direction1[2] << " " << std::endl;
      //std::cout << "SlowdonwDir perp" << perp_direction2[0] << " " << perp_direction2[1] << " " << perp_direction2[2] << " " << std::endl;
      //SFT
      float ti_eV = interp2dCombined(x, y, z, nR_Temp, nZ_Temp, TempGridr, TempGridz, ti);
      float density = interp2dCombined(x, y, z, nR_Dens, nZ_Dens, DensGridr, DensGridz, ni);
      float tau_s = particlesPointer->amu[indx] * ti_eV * std::sqrt(ti_eV / background_amu) / (6.84e4 * (1.0 + background_amu / particlesPointer->amu[indx]) * density / 1.0e18 * particlesPointer->charge[indx] * particlesPointer->charge[indx] * 15);
      float tau_par = particlesPointer->amu[indx] * ti_eV * std::sqrt(ti_eV / background_amu) / (6.84e4 * density / 1.0e18 * particlesPointer->charge[indx] * particlesPointer->charge[indx] * 15);
      float tau_E = particlesPointer->amu[indx] * ti_eV * std::sqrt(ti_eV / background_amu) / (1.4e5 * density / 1.0e18 * particlesPointer->charge[indx] * particlesPointer->charge[indx] * 15);
      //std::cout << "tau_E " << tau_E << endl;
      //std::cout << "ti dens tau_s tau_par " << ti_eV << " " << density << " " << tau_s << " " << tau_par << endl;
      float vTherm = std::sqrt(ti_eV * 1.602e-19 / particlesPointer->amu[indx] / 1.66e-27);

      //std::cout << "vTherm tau_s " << vTherm << " " << tau_s  << endl;
      //        float vxy00 = sqrt(vTherm*vTherm - vz*vz);
      //        float vxy01 = sqrt(vx*vx + vy*vy);
      //	//std::cout << "vzNew vxy0 vxy " << vzNew << " " << vxy0 << " " << vxy << endl;
      //        particlesPointer->vx[indx] = vx/vxy01*vxy00;///velocityCollisionsNorm;
      //	particlesPointer->vy[indx] = vy/vxy01*vxy00;///velocityCollisionsNorm;
      //particlesPointer->vz[indx] = vzNew;///velocityCollisionsNorm;
      float drag = -dt * nu_friction * velocityRelativeNorm / 1.2;
      //SFT
      //particlesPointer->nu_s[indx]=nu_friction;
      //particlesPointer->vD[indx]=flowVelocity[2];
      //drag = -dt/tau_s*velocityRelativeNorm;
      //std::cout << "nu_friction " << nu_friction << std::endl;
      //std::cout << "nu_paralel " << nu_parallel << std::endl;
      //std::cout << "nu_deflection " << nu_deflection << std::endl;
      //std::cout << "nu_energy " << nu_energy << std::endl;
      //std::cout << "nu_paralel base " << sqrt(0.5*nu_parallel*dt*vRel2) << std::endl;
      //std::cout << "vrelative " << velocityRelativeNorm << std::endl;
      //std::cout << "particle z " << particlesPointer->z[indx] << std::endl;
      //std::cout << "vRel2 " << vRel2 << std::endl;
      //std::cout << "nu_energy " << nu_energy << std::endl;
      //std::cout << "vTherm2 " << vTherm*vTherm << std::endl;
      //std::cout << "drag " << drag << std::endl;
      float coeff_par = 1.4142 * n1 * std::sqrt(nu_parallel * dt);
      //SFT
      //coeff_par = n1*sqrt(dt/tau_par);
      //int plumin3 = 2*floor(r3+0.5) - 1;
      //float coeff_par = plumin3*sqrt(0.5*abs(nu_energy)*dt*vRel2);
      //float coeff_par = 1.0;//+ n1*sqrt(nu_parallel*dt);
      //float coeff_par = 1.0 - nu_friction*dt;
      //int plumin1 = 2*floor(r1+0.5) - 1;
      //int plumin2 = 2*floor(r2+0.5) - 1;
      float cosXsi = cos(2.0 * pi * xsi);
      float sinXsi = sin(2.0 * pi * xsi);
      float coeff_perp1 = cosXsi * std::sqrt(nu_deflection * dt * 0.5);
      float coeff_perp2 = sinXsi * std::sqrt(nu_deflection * dt * 0.5);
//std::cout << "cosXsi and sinXsi " << cosXsi << " " << sinXsi << std::endl;
#if USEFRICTION == 0
      drag = 0.0;
#endif
#if USEANGLESCATTERING == 0
      coeff_perp1 = 0.0;
      coeff_perp2 = 0.0;
#endif
#if USEHEATING == 0
      coeff_par = 0.0;
#endif
      ////ALL COULOMB COLLISION OPERATORS///
      velocityCollisions[0] = (drag)*relativeVelocity[0] / velocityRelativeNorm;
      velocityCollisions[1] = (drag)*relativeVelocity[1] / velocityRelativeNorm;
      velocityCollisions[2] = (drag)*relativeVelocity[2] / velocityRelativeNorm;
      ////ALL COULOMB COLLISION OPERATORS///
      //velocityCollisions[0] = (drag*parallel_direction[0] + coeff_perp1*perp_direction1[0] + coeff_perp2*perp_direction2[0]);
      //velocityCollisions[1] = (drag*parallel_direction[1] + coeff_perp1*perp_direction1[1] + coeff_perp2*perp_direction2[1]);
      //velocityCollisions[2] = (drag*parallel_direction[2] + coeff_perp1*perp_direction1[2] + coeff_perp2*perp_direction2[2]);
      //float velocityColl = vectorNorm(velocityCollisions);
      //velocityCollisions[0] = coeff_par*velocityCollisions[0]/velocityColl;
      //velocityCollisions[1] = coeff_par*velocityCollisions[1]/velocityColl;
      //velocityCollisions[2] = coeff_par*velocityCollisions[2]/velocityColl;
      ////HEATING ONLY //////
      //velocityCollisions[0] = (coeff_par)*parallel_direction[0];
      //velocityCollisions[1] = (coeff_par)*parallel_direction[1];
      //velocityCollisions[2] = (coeff_par)*parallel_direction[2];
      float velocityCollisionsNorm = vectorNorm(velocityCollisions);
      //vUpdate[0] = velocityRelativeNorm*velocityCollisions[0] + flowVelocity[0];
      //vUpdate[1] = velocityRelativeNorm*velocityCollisions[1] + flowVelocity[1];
      //vUpdate[2] = velocityRelativeNorm*velocityCollisions[2] + flowVelocity[2];
      //float velocityColl = vectorNorm(vUpdate);

      //particlesPointer->vx[indx] = coeff_Energy*vPartNorm*vUpdate[0]/velocityColl;
      //particlesPointer->vy[indx] = coeff_Energy*vPartNorm*vUpdate[1]/velocityColl;
      //particlesPointer->vz[indx] = coeff_Energy*vPartNorm*vUpdate[2]/velocityColl;
      //vPartNorm = sqrt(vfx*vfx + vfy*vfy + vfz*vfz);
      //float vzNew = vz + drag*std::copysign(1.0,parallel_direction[2]);///velocityCollisionsNorm;

      //float vxy0 = sqrt(vx*vx + vy*vy);
      //float vxy = sqrt(vPartNorm*vPartNorm - vzNew*vzNew);
      ////std::cout << "vzNew vxy0 vxy " << vzNew << " " << vxy0 << " " << vxy << endl;
      //particlesPointer->vx[indx] = vx/vxy0*vxy;///velocityCollisionsNorm;
      //particlesPointer->vy[indx] = vy/vxy0*vxy;///velocityCollisionsNorm;
      //particlesPointer->vz[indx] = vzNew;///velocityCollisionsNorm;
      //if(1.0-0.5*nu_energy*dt > 1.5){
      //std::cout << " vpartNorm nu_energy " << vPartNorm << " " <<nu_energy<< std::endl;
      //std::cout << "vpartnorm ecoeff coeffpar vel coll " << vPartNorm << " " << (1.0-0.5*nu_energy*dt) << " " << (1+coeff_par) << " "
      //std::cout << "velocity collisions " <<velocityCollisions[0] << " " << velocityCollisions[1] << " " << velocityCollisions[2] << std::endl;
      //nu_energy = 0.0;
      //}
      //if(1.0-0.5*nu_energy*dt < 0.5){
      //std::cout << "vPartNorm nu_energy small " <<vPartNorm << " " <<  nu_energy<< std::endl;
      //nu_energy = 0.0;
      //}
      float nuEdt = nu_energy * dt;
      if (nuEdt < -1.0)
        nuEdt = -1.0;
      particlesPointer->vx[indx] = vPartNorm * (1.0 - 0.5 * nuEdt) * ((1 + coeff_par) * parallel_direction[0] + std::abs(n2) * (coeff_perp1 * perp_direction1[0] + coeff_perp2 * perp_direction2[0])) + velocityCollisions[0];
      particlesPointer->vy[indx] = vPartNorm * (1.0 - 0.5 * nuEdt) * ((1 + coeff_par) * parallel_direction[1] + std::abs(n2) * (coeff_perp1 * perp_direction1[1] + coeff_perp2 * perp_direction2[1])) + velocityCollisions[1];
      particlesPointer->vz[indx] = vPartNorm * (1.0 - 0.5 * nuEdt) * ((1 + coeff_par) * parallel_direction[2] + std::abs(n2) * (coeff_perp1 * perp_direction1[2] + coeff_perp2 * perp_direction2[2])) + velocityCollisions[2];
      //SFT
      //float tauE_dt = tau_E*dt;
      //if(tauE_dt < -1.0) tauE_dt = -1.0;
      //particlesPointer->vx[indx] = vPartNorm*(1.0-0.5*tauE_dt)*((1+coeff_par)*parallel_direction[0] + n2*(coeff_perp1*perp_direction1[0] + coeff_perp2*perp_direction2[0])) + velocityCollisions[0];
      //particlesPointer->vy[indx] = vPartNorm*(1.0-0.5*tauE_dt)*((1+coeff_par)*parallel_direction[1] + n2*(coeff_perp1*perp_direction1[1] + coeff_perp2*perp_direction2[1])) + velocityCollisions[1];
      //particlesPointer->vz[indx] = vPartNorm*(1.0-0.5*tauE_dt)*((1+coeff_par)*parallel_direction[2] + n2*(coeff_perp1*perp_direction1[2] + coeff_perp2*perp_direction2[2])) + velocityCollisions[2];
      //particlesPointer->vx[indx] = vPartNorm*(1+coeff_par )*parallel_direction[0] + velocityCollisions[0];
      //particlesPointer->vy[indx] = vPartNorm*(1+coeff_par )*parallel_direction[1] + velocityCollisions[1];
      //particlesPointer->vz[indx] = vPartNorm*(1+coeff_par )*parallel_direction[2] + velocityCollisions[2];

      vx = particlesPointer->vx[indx];
      vy = particlesPointer->vy[indx];
      vz = particlesPointer->vz[indx];

      //if(vNew > 1.0e5){
      //std::cout << "vpartnorm ecoeff coeffpar vel coll " << vPartNorm << " " << (1.0-0.5*nu_energy*dt) << " " << (1+coeff_par) << " "
      //<<velocityCollisions[0] << std::endl;
      //}
      //if(isnan(particlesPointer->vx[indx]))
      //{
      //    std::cout << "nu_d " << nu_deflection<< std::endl;
      //std::cout << "nu_energy oeff " << (1.0-nu_energy*0.5*dt)<< std::endl;
      //    std::cout << "vxvyvz" << vx<< " " << vy << " " << vz<< " "  << std::endl;
      //    //std::cout << "vf isnan" << vfx<< " " << vPartNorm << " " << vbx<< " " << coeff_perp << std::endl;
      //std::cout << "particle velocity " << particlesPointer->vx[indx] << " " << particlesPointer->vy[indx] << " " << particlesPointer->vz[indx] << std::endl;
      //std::cout << "velocityCollisions " << velocityCollisions[0] << " " << velocityCollisions[1] << " " << velocityCollisions[2] << std::endl;
      // std::cout << "SlowdonwDir par" << parallel_direction[0] << " " << parallel_direction[1] << " " << parallel_direction[2] << " " << std::endl;
      // std::cout << "SlowdonwDir perp" << perp_direction1[0] << " " <<perp_direction1[1] << " " << perp_direction1[2] << " " << std::endl;
      // std::cout << "SlowdonwDir perpi2" << perp_direction2[0] << " " << perp_direction2[1] << " " << perp_direction2[2] << " " << std::endl;
      //}
      //particlesPointer->vx[indx] = vb;
      //particlesPointer->vy[indx] = vf;
      //particlesPointer->vz[indx] = vf;
      //particlesPointer->vx[indx] = vPartNorm*velocityCollisions[0];
      //particlesPointer->vy[indx] = vPartNorm*velocityCollisions[1];
      //particlesPointer->vz[indx] = vPartNorm*velocityCollisions[2];

      this->dv[0] = velocityCollisions[0];
      this->dv[1] = velocityCollisions[1];
      this->dv[2] = velocityCollisions[2];
      //if(particlesPointer->vx[indx] != particlesPointer->vx[indx])
      //{
      //    if(particlesPointer->test[indx] == 0.0)
      //    {
      //        particlesPointer->test[indx] = 1.0;
      //        particlesPointer->test0[indx]  = nu_energy;
      //        particlesPointer->test1[indx] = flowVelocity[0];
      //        particlesPointer->test2[indx] = flowVelocity[1];
      //        particlesPointer->test3[indx] = flowVelocity[2];
      //    }
      //}
      //if( nu_parallel > particlesPointer->test1[indx])
      //{
      //    particlesPointer->test1[indx] =  nu_parallel;
      //    particlesPointer->test2[indx] =  nu_energy;
      //    particlesPointer->test3[indx] =  nu_friction;
      //    particlesPointer->test4[indx] =  nu_deflection;
      //}
      //        particlesPointer->test[indx] =  vUpdate[2];
      //        particlesPointer->test0[indx] = particlesPointer->vz[indx];//velocityRelativeNorm*velocityCollisions[1]+ flowVelocity[1];
      //        particlesPointer->test1[indx] = velocityCollisionsNorm;//flowVelocity[0];
      //        particlesPointer->test2[indx] = velocityCollisions[2];
      //        particlesPointer->test3[indx] = flowVelocity[2];
      //std::cout << "particle velocity "<< particlesPointer->vx[indx] << " " << particlesPointer->vy[indx] << " " << particlesPointer->vz[indx]    << std::endl;
    }
  }
};

#endif
