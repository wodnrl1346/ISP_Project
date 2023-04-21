""" homomorphic filtering
1. zero padding
2. ln
3. centering
4. DFT
5. Filtering
6. IDFT
7. de-centering
8. exp
9. remove zero-padding
centering을 하면 spatial domain에서의 값들이 변하기 때문에 centering 이전에 ln을 취해야한다.
"""

# Frequency domain filtering을 위한 module import
import cv2
import matplotlib.pyplot as plt
import numpy as np

# image를 gray-scale로 읽어온다.
image = cv2.imread("moon.png", 0)
plt.imshow(image, cmap='gray'), plt.axis('off')
plt.title("Original Image")
plt.show()

# step1: zero-padding
M, N = image.shape
P, Q = M * 2, N * 2
padded_image = np.zeros((P, Q))
padded_image[:M, :N] = image

# zero-padding한 결과를 보여준다.
plt.imshow(padded_image, cmap='gray'), plt.axis('off')
plt.title("step 1. zero-padding")
plt.show()

# step2(ln): ln 로그를 취해서 illumination과 reflectance를 분리해준다.
# zero-padding된 부분은 log를 그대로 씌우면 안된다(devide by zero encountered in log 오류가 발생) 로그의 진수는 양수여야 하기 때문이다.
# 따라서 이를 해결하기 위해 np.log()대신 np.log(+1)를 사용했다.
z = np.log(padded_image+1)

# padded_image에 ln연산한 결과를 보여준다.
plt.imshow(z, cmap='gray'), plt.axis('off')
plt.title("step 2. ln")
plt.show()

# step3(centering): Filtering을 하기 위해 저주파 성분을 center로 모은다.
z_new = np.zeros((P, Q))
for x in range(P):
    for y in range(Q):
        z_new[x, y] = z[x, y] * ((-1)**(x+y))

# step4(DFT): np.fft.fft2() 함수를 취하여 DFT하고, 주파수 영역에서 해석한다. Z(u,v)
# 이를 logscale으로 spectrum을 표현했다.
Z = np.fft.fft2(z_new)
plt.imshow(np.log(np.abs(Z)).real, cmap='gray'), plt.axis('off')
plt.title("step 3~4. centering + DFT")
plt.show()

# step5(H(u,v)): homomorphic filtering을 구현하고 이를 Z(u, v)와 곱하여 S(u,v)=H(u,v)*Z(u,v)를 구한다.
# illumination은 저주파이고, reflectance는 고주파이다. 따라서 rL은 illumination을, rH은 reflectance를 조절하는 parameter이다.
# 다른 값들을 고정하고 rL을 변화시켰을때, rL를 낮추면 더 어두워지고, rL을 높이면 더 밝아지는 것을 알 수 있었다.
# 다른 값들을 고정하고 rH을 변화시켰을때, rH를 낮추면 detail이 줄어들고, rH를 높이면 detail이 많이 표현되는 것을 알 수 있다.

def Homomorphic_filter(image, c, cutoff, rL, rH):
    M, N = image.shape
    H = np.zeros((M, N))

    # image의 center
    U0 = int(M/2)
    V0 = int(N/2)

    # cutoff frequency
    D0 = cutoff

    # homomorphic filter 식 구현
    for u in range(M):
        for v in range(N):
            u_ = (np.abs(u-U0))**2
            v_ = (np.abs(v-V0))**2
            H[u, v] = ((rH-rL)*(1 - np.exp(-c*(((np.sqrt(u_+v_))**2)/(D0**2))))) + rL

    return H


# function을 취할 함수와 parameter를 지정한다.
# trial & error를 통해 이 image에 적합한 parameter를 구했다.
# 그 결과 c=1, cutoff=450, rL=0.8, rH=1.8 정도에서 조명을 줄이고 detail이 살아 나는 것을 알 수 있었다.
# 자세히 보면, 달 둘레 주위로 띠가 보이는 것을 확인할 수 있다. 또한, 보이지 않는 달의 반쪽에서도 detail이 어느정도 살아났다.
# 달 표면의 detail도 확인할 수 있었다.
H = Homomorphic_filter(Z, c=1, cutoff=450, rL=0.8, rH=1.5)
plt.imshow(H.real, cmap='gray'), plt.axis('off')
plt.title("step 5. Homomorphic Filter")
plt.show()

# S(u, v) = H(u,v) * Z(u,v), homomorphic filtering 한 결과
S = np.multiply(H, Z)
plt.imshow(np.log(np.abs(S)).real, cmap='gray'), plt.axis('off')
plt.title("step 5. Homomorphic Filtered result")
plt.show()

# step6(IDFT): step5에서 구한 S(u,v)를 Inverse DFT하여 s(x,y)를 얻는다.
s = np.fft.ifft2(S)

# step7: step3에서 filtering을 위해 진행한 centering을 상쇄하는 de-centering
for x in range(P):
    for y in range(Q):
        s[x, y] = s[x, y] * ((-1)**(x+y))

# step8(exp): s(x, y)에 exp을 취하여 step2의 ln을 상쇄시킨다.
g = np.exp(s)

# step9: step1에서 진행한 zero-padding part를 remove한다.
g = g[:M, :N]

img_homo = g.real

# homomorphic filtering된 결과를 보여준다.
plt.imshow(img_homo, cmap='gray'), plt.axis('off')
plt.title("step 6~9. Final Result")
plt.show()


'''Laplacian'''
def Laplacian(image):
    M, N = image.shape
    H = np.zeros((M, N))

    for u in range(M):
        for v in range(N):
            u2 = np.power(u - M/2, 2)
            v2 = np.power(v - N/2, 2)
            H[u, v] = -(u2+v2)

    return H


# step1: zero-padding
M, N = img_homo.shape
P, Q = M * 2, N * 2
padded_image = np.zeros((P, Q))
padded_image[:M, :N] = img_homo

# step2(centering): Filtering을 하기 위해 저주파 성분을 center로 모은다.
z_new = np.zeros((P, Q))
for x in range(P):
    for y in range(Q):
        z_new[x, y] = z[x, y] * ((-1)**(x+y))

# step3(DFT): np.fft.fft2() 함수를 취하여 DFT하고, 주파수 영역에서 해석한다. Z(u,v)
Z = np.fft.fft2(z_new)

# step4(H(u,v)): Laplacian
H = Laplacian(Z)
S = np.multiply(H, Z)

# step5(IDFT): step5에서 구한 S(u,v)를 Inverse DFT하여 s(x,y)를 얻는다.
s = np.fft.ifft2(S)

# step6: de-centering
for x in range(P):
    for y in range(Q):
        s[x, y] = s[x, y] * ((-1)**(x+y))

# step7: step1에서 진행한 zero-padding part를 remove한다.
s = s[:M, :N]
min, max = np.min(s), np.max(s)
s = (s - min) / (max - min) * 255.0

img_laplacian = s.real

plt.imshow(img_laplacian, cmap='gray'), plt.axis('off')
plt.title("Laplcian")
plt.show()

plt.imshow(img_homo+img_laplacian, cmap='gray'), plt.axis('off')
plt.title("Homo+Laplcian Result")
plt.show()