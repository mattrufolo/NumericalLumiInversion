### Reason behind the computation

From [Herr and Muratori](https://cds.cern.ch/record/941318/files/p361.pdf), assuming that the beams distributions are factorizable in the three spatial directions and a particular dependence between the axes, it is possible to obtain this formula for the instantaneous luminosity

$$
    L = N_1N_2n_bf_r\sqrt{(\vec{v_1}-\vec{v_2})^2-\frac{\vec{v_1}\times\vec{v_2}}{c^2}}\int_{-\infty}^\infty\int_{-\infty}^\infty\rho_z^{B1}\rho_z^{B2}dt \int_{-\infty}^\infty\rho_x^{B1}\rho_x^{B2}dx\int_{-\infty}^\infty\rho_y^{B1}\rho_y^{B2}dydz
$$

Assuming that the beam along the three directions have a Gaussian distribution:

$$
        \rho^B_i = \frac{1}{\sqrt{2\pi}\sigma_{i,B}}e^{-\frac{(i-\mu_{i,B})^2}{2\sigma_{i,B}^2}}, \quad \text{with } B\in\{B1,B2\} \text{ and } i\in\{x,y,z\}
$$

where 

From this assumption and doing some computation, it is possible to rewrite the luminosity formula like: 

$$
    L = N_1N_2n_bf_r\frac{\sqrt{{c^2}(\vec{v_1}-\vec{v_2})^2-\vec{v_1}\times\vec{v_2}}}{2c^2}\int_{-\infty}^\infty \frac{e^{\frac{-(\mu_{x1}-\mu_{x2})^2}{2\beta(z)(\epsilon_{x1}+\epsilon_{x2})}}e^{\frac{-(\mu_{y1}-\mu_{y2})^2}{2\beta(z)(\epsilon_{y1}+\epsilon_{y2})}}}{\sqrt{2}\pi^{\frac{3}{2}}\sqrt{(\sigma_{x1}^2+\sigma_{x2}^2)}\sqrt{(\sigma_{y1}^2+\sigma_{y2}^2)}}\rho_z^{B1}\rho_z^{B2}dz
$$

where if we approximate  $\sigma_{u} = \beta(z)\epsilon_{u},  \text{ where } u \in ${x1,x2,y1,y2}, it is possible to obtain

$$
     L  =
         C\int_{-\infty}^\infty \frac{e^{\frac{-(\mu_{x1}-\mu_{x2})^2}{2\beta(z)(\epsilon_{x1}+\epsilon_{x2})}}e^{\frac{-(\mu_{y1}-\mu_{y2})^2}{2\beta(z)(\epsilon_{y1}+\epsilon_{y2})}}}{\sqrt{2}\pi^{\frac{3}{2}}\beta(z)\sqrt{(\epsilon_{x1}+\epsilon_{x2})(\epsilon_{y1}+\epsilon_{y2})}}\rho_z^{B1}\rho_z^{B2}dz  \nonumber \\
     = L((\epsilon_{x1}+\epsilon_{x2}),(\epsilon_{y1}+\epsilon_{y2})|\dots)
$$


