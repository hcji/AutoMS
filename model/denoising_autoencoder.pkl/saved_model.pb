²
ò
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype

Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	

Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
?
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:

RandomStandardNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
@
ReadVariableOp
resource
value"dtype"
dtypetype
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
Á
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
÷
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.9.02unknown8µÏ

Adam/dense_35/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*%
shared_nameAdam/dense_35/bias/v
y
(Adam/dense_35/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_35/bias/v*
_output_shapes
:2*
dtype0

Adam/dense_35/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: 2*'
shared_nameAdam/dense_35/kernel/v

*Adam/dense_35/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_35/kernel/v*
_output_shapes

: 2*
dtype0

Adam/dense_34/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/dense_34/bias/v
y
(Adam/dense_34/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_34/bias/v*
_output_shapes
: *
dtype0

Adam/dense_34/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *'
shared_nameAdam/dense_34/kernel/v

*Adam/dense_34/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_34/kernel/v*
_output_shapes

:  *
dtype0

Adam/dense_33/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/dense_33/bias/v
y
(Adam/dense_33/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_33/bias/v*
_output_shapes
: *
dtype0

Adam/dense_33/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *'
shared_nameAdam/dense_33/kernel/v

*Adam/dense_33/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_33/kernel/v*
_output_shapes

:@ *
dtype0

Adam/latent_vector/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@**
shared_nameAdam/latent_vector/bias/v

-Adam/latent_vector/bias/v/Read/ReadVariableOpReadVariableOpAdam/latent_vector/bias/v*
_output_shapes
:@*
dtype0

Adam/latent_vector/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	¨@*,
shared_nameAdam/latent_vector/kernel/v

/Adam/latent_vector/kernel/v/Read/ReadVariableOpReadVariableOpAdam/latent_vector/kernel/v*
_output_shapes
:	¨@*
dtype0

Adam/conv1d_23/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv1d_23/bias/v
{
)Adam/conv1d_23/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_23/bias/v*
_output_shapes
:*
dtype0

Adam/conv1d_23/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv1d_23/kernel/v

+Adam/conv1d_23/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_23/kernel/v*"
_output_shapes
:*
dtype0

Adam/conv1d_22/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv1d_22/bias/v
{
)Adam/conv1d_22/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_22/bias/v*
_output_shapes
:*
dtype0

Adam/conv1d_22/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv1d_22/kernel/v

+Adam/conv1d_22/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_22/kernel/v*"
_output_shapes
:*
dtype0

Adam/dense_35/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*%
shared_nameAdam/dense_35/bias/m
y
(Adam/dense_35/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_35/bias/m*
_output_shapes
:2*
dtype0

Adam/dense_35/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: 2*'
shared_nameAdam/dense_35/kernel/m

*Adam/dense_35/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_35/kernel/m*
_output_shapes

: 2*
dtype0

Adam/dense_34/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/dense_34/bias/m
y
(Adam/dense_34/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_34/bias/m*
_output_shapes
: *
dtype0

Adam/dense_34/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *'
shared_nameAdam/dense_34/kernel/m

*Adam/dense_34/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_34/kernel/m*
_output_shapes

:  *
dtype0

Adam/dense_33/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/dense_33/bias/m
y
(Adam/dense_33/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_33/bias/m*
_output_shapes
: *
dtype0

Adam/dense_33/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *'
shared_nameAdam/dense_33/kernel/m

*Adam/dense_33/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_33/kernel/m*
_output_shapes

:@ *
dtype0

Adam/latent_vector/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@**
shared_nameAdam/latent_vector/bias/m

-Adam/latent_vector/bias/m/Read/ReadVariableOpReadVariableOpAdam/latent_vector/bias/m*
_output_shapes
:@*
dtype0

Adam/latent_vector/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	¨@*,
shared_nameAdam/latent_vector/kernel/m

/Adam/latent_vector/kernel/m/Read/ReadVariableOpReadVariableOpAdam/latent_vector/kernel/m*
_output_shapes
:	¨@*
dtype0

Adam/conv1d_23/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv1d_23/bias/m
{
)Adam/conv1d_23/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_23/bias/m*
_output_shapes
:*
dtype0

Adam/conv1d_23/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv1d_23/kernel/m

+Adam/conv1d_23/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_23/kernel/m*"
_output_shapes
:*
dtype0

Adam/conv1d_22/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv1d_22/bias/m
{
)Adam/conv1d_22/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_22/bias/m*
_output_shapes
:*
dtype0

Adam/conv1d_22/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv1d_22/kernel/m

+Adam/conv1d_22/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_22/kernel/m*"
_output_shapes
:*
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
r
dense_35/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*
shared_namedense_35/bias
k
!dense_35/bias/Read/ReadVariableOpReadVariableOpdense_35/bias*
_output_shapes
:2*
dtype0
z
dense_35/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: 2* 
shared_namedense_35/kernel
s
#dense_35/kernel/Read/ReadVariableOpReadVariableOpdense_35/kernel*
_output_shapes

: 2*
dtype0
r
dense_34/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_34/bias
k
!dense_34/bias/Read/ReadVariableOpReadVariableOpdense_34/bias*
_output_shapes
: *
dtype0
z
dense_34/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  * 
shared_namedense_34/kernel
s
#dense_34/kernel/Read/ReadVariableOpReadVariableOpdense_34/kernel*
_output_shapes

:  *
dtype0
r
dense_33/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_33/bias
k
!dense_33/bias/Read/ReadVariableOpReadVariableOpdense_33/bias*
_output_shapes
: *
dtype0
z
dense_33/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ * 
shared_namedense_33/kernel
s
#dense_33/kernel/Read/ReadVariableOpReadVariableOpdense_33/kernel*
_output_shapes

:@ *
dtype0
|
latent_vector/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*#
shared_namelatent_vector/bias
u
&latent_vector/bias/Read/ReadVariableOpReadVariableOplatent_vector/bias*
_output_shapes
:@*
dtype0

latent_vector/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	¨@*%
shared_namelatent_vector/kernel
~
(latent_vector/kernel/Read/ReadVariableOpReadVariableOplatent_vector/kernel*
_output_shapes
:	¨@*
dtype0
t
conv1d_23/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv1d_23/bias
m
"conv1d_23/bias/Read/ReadVariableOpReadVariableOpconv1d_23/bias*
_output_shapes
:*
dtype0

conv1d_23/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv1d_23/kernel
y
$conv1d_23/kernel/Read/ReadVariableOpReadVariableOpconv1d_23/kernel*"
_output_shapes
:*
dtype0
t
conv1d_22/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv1d_22/bias
m
"conv1d_22/bias/Read/ReadVariableOpReadVariableOpconv1d_22/bias*
_output_shapes
:*
dtype0

conv1d_22/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv1d_22/kernel
y
$conv1d_22/kernel/Read/ReadVariableOpReadVariableOpconv1d_22/kernel*"
_output_shapes
:*
dtype0

NoOpNoOp
Íb
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*b
valueþaBûa Bôa
§
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*	&call_and_return_all_conditional_losses

_default_save_signature
	optimizer

signatures*
* 
¬
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*

layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer-4
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses*
Z
#0
$1
%2
&3
'4
(5
)6
*7
+8
,9
-10
.11*
Z
#0
$1
%2
&3
'4
(5
)6
*7
+8
,9
-10
.11*
* 
°
/non_trainable_variables

0layers
1metrics
2layer_regularization_losses
3layer_metrics
	variables
trainable_variables
regularization_losses
__call__

_default_save_signature
*	&call_and_return_all_conditional_losses
&	"call_and_return_conditional_losses*
6
4trace_0
5trace_1
6trace_2
7trace_3* 
6
8trace_0
9trace_1
:trace_2
;trace_3* 
* 
´
<iter

=beta_1

>beta_2
	?decay
@learning_rate#mØ$mÙ%mÚ&mÛ'mÜ(mÝ)mÞ*mß+mà,má-mâ.mã#vä$vå%væ&vç'vè(vé)vê*vë+vì,ví-vî.vï*

Aserving_default* 

B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
F__call__
*G&call_and_return_all_conditional_losses* 
È
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
L__call__
*M&call_and_return_all_conditional_losses

#kernel
$bias
 N_jit_compiled_convolution_op*
È
O	variables
Ptrainable_variables
Qregularization_losses
R	keras_api
S__call__
*T&call_and_return_all_conditional_losses

%kernel
&bias
 U_jit_compiled_convolution_op*

V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
Z__call__
*[&call_and_return_all_conditional_losses* 
¦
\	variables
]trainable_variables
^regularization_losses
_	keras_api
`__call__
*a&call_and_return_all_conditional_losses

'kernel
(bias*
.
#0
$1
%2
&3
'4
(5*
.
#0
$1
%2
&3
'4
(5*
* 

bnon_trainable_variables

clayers
dmetrics
elayer_regularization_losses
flayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
gtrace_0
htrace_1
itrace_2
jtrace_3* 
6
ktrace_0
ltrace_1
mtrace_2
ntrace_3* 
* 
¦
o	variables
ptrainable_variables
qregularization_losses
r	keras_api
s__call__
*t&call_and_return_all_conditional_losses

)kernel
*bias*
¦
u	variables
vtrainable_variables
wregularization_losses
x	keras_api
y__call__
*z&call_and_return_all_conditional_losses

+kernel
,bias*
§
{	variables
|trainable_variables
}regularization_losses
~	keras_api
__call__
+&call_and_return_all_conditional_losses

-kernel
.bias*

	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses* 
.
)0
*1
+2
,3
-4
.5*
.
)0
*1
+2
,3
-4
.5*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses*
:
trace_0
trace_1
trace_2
trace_3* 
:
trace_0
trace_1
trace_2
trace_3* 
PJ
VARIABLE_VALUEconv1d_22/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEconv1d_22/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEconv1d_23/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEconv1d_23/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUElatent_vector/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUElatent_vector/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEdense_33/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEdense_33/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEdense_34/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEdense_34/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEdense_35/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEdense_35/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
1
2*

0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses* 

trace_0* 

trace_0* 

#0
$1*

#0
$1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
 layer_metrics
H	variables
Itrainable_variables
Jregularization_losses
L__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses*

¡trace_0* 

¢trace_0* 
* 

%0
&1*

%0
&1*
* 

£non_trainable_variables
¤layers
¥metrics
 ¦layer_regularization_losses
§layer_metrics
O	variables
Ptrainable_variables
Qregularization_losses
S__call__
*T&call_and_return_all_conditional_losses
&T"call_and_return_conditional_losses*

¨trace_0* 

©trace_0* 
* 
* 
* 
* 

ªnon_trainable_variables
«layers
¬metrics
 ­layer_regularization_losses
®layer_metrics
V	variables
Wtrainable_variables
Xregularization_losses
Z__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses* 

¯trace_0* 

°trace_0* 

'0
(1*

'0
(1*
* 

±non_trainable_variables
²layers
³metrics
 ´layer_regularization_losses
µlayer_metrics
\	variables
]trainable_variables
^regularization_losses
`__call__
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses*

¶trace_0* 

·trace_0* 
* 
.
0
1
2
3
4
5*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

)0
*1*

)0
*1*
* 

¸non_trainable_variables
¹layers
ºmetrics
 »layer_regularization_losses
¼layer_metrics
o	variables
ptrainable_variables
qregularization_losses
s__call__
*t&call_and_return_all_conditional_losses
&t"call_and_return_conditional_losses*

½trace_0* 

¾trace_0* 

+0
,1*

+0
,1*
* 

¿non_trainable_variables
Àlayers
Ámetrics
 Âlayer_regularization_losses
Ãlayer_metrics
u	variables
vtrainable_variables
wregularization_losses
y__call__
*z&call_and_return_all_conditional_losses
&z"call_and_return_conditional_losses*

Ätrace_0* 

Åtrace_0* 

-0
.1*

-0
.1*
* 

Ænon_trainable_variables
Çlayers
Èmetrics
 Élayer_regularization_losses
Êlayer_metrics
{	variables
|trainable_variables
}regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*

Ëtrace_0* 

Ìtrace_0* 
* 
* 
* 

Ínon_trainable_variables
Îlayers
Ïmetrics
 Ðlayer_regularization_losses
Ñlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 

Òtrace_0* 

Ótrace_0* 
* 
'
0
1
2
3
4*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
Ô	variables
Õ	keras_api

Ötotal

×count*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

Ö0
×1*

Ô	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/conv1d_22/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/conv1d_22/bias/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/conv1d_23/kernel/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/conv1d_23/bias/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUEAdam/latent_vector/kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUEAdam/latent_vector/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/dense_33/kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/dense_33/bias/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/dense_34/kernel/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/dense_34/bias/mBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/dense_35/kernel/mCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/dense_35/bias/mCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/conv1d_22/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/conv1d_22/bias/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/conv1d_23/kernel/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/conv1d_23/bias/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUEAdam/latent_vector/kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUEAdam/latent_vector/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/dense_33/kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/dense_33/bias/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/dense_34/kernel/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/dense_34/bias/vBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/dense_35/kernel/vCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/dense_35/bias/vCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

serving_default_encoder_inputPlaceholder*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
dtype0* 
shape:ÿÿÿÿÿÿÿÿÿ2
¢
StatefulPartitionedCallStatefulPartitionedCallserving_default_encoder_inputconv1d_22/kernelconv1d_22/biasconv1d_23/kernelconv1d_23/biaslatent_vector/kernellatent_vector/biasdense_33/kerneldense_33/biasdense_34/kerneldense_34/biasdense_35/kerneldense_35/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference_signature_wrapper_862683
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv1d_22/kernel/Read/ReadVariableOp"conv1d_22/bias/Read/ReadVariableOp$conv1d_23/kernel/Read/ReadVariableOp"conv1d_23/bias/Read/ReadVariableOp(latent_vector/kernel/Read/ReadVariableOp&latent_vector/bias/Read/ReadVariableOp#dense_33/kernel/Read/ReadVariableOp!dense_33/bias/Read/ReadVariableOp#dense_34/kernel/Read/ReadVariableOp!dense_34/bias/Read/ReadVariableOp#dense_35/kernel/Read/ReadVariableOp!dense_35/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp+Adam/conv1d_22/kernel/m/Read/ReadVariableOp)Adam/conv1d_22/bias/m/Read/ReadVariableOp+Adam/conv1d_23/kernel/m/Read/ReadVariableOp)Adam/conv1d_23/bias/m/Read/ReadVariableOp/Adam/latent_vector/kernel/m/Read/ReadVariableOp-Adam/latent_vector/bias/m/Read/ReadVariableOp*Adam/dense_33/kernel/m/Read/ReadVariableOp(Adam/dense_33/bias/m/Read/ReadVariableOp*Adam/dense_34/kernel/m/Read/ReadVariableOp(Adam/dense_34/bias/m/Read/ReadVariableOp*Adam/dense_35/kernel/m/Read/ReadVariableOp(Adam/dense_35/bias/m/Read/ReadVariableOp+Adam/conv1d_22/kernel/v/Read/ReadVariableOp)Adam/conv1d_22/bias/v/Read/ReadVariableOp+Adam/conv1d_23/kernel/v/Read/ReadVariableOp)Adam/conv1d_23/bias/v/Read/ReadVariableOp/Adam/latent_vector/kernel/v/Read/ReadVariableOp-Adam/latent_vector/bias/v/Read/ReadVariableOp*Adam/dense_33/kernel/v/Read/ReadVariableOp(Adam/dense_33/bias/v/Read/ReadVariableOp*Adam/dense_34/kernel/v/Read/ReadVariableOp(Adam/dense_34/bias/v/Read/ReadVariableOp*Adam/dense_35/kernel/v/Read/ReadVariableOp(Adam/dense_35/bias/v/Read/ReadVariableOpConst*8
Tin1
/2-	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *(
f#R!
__inference__traced_save_863445
¦	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv1d_22/kernelconv1d_22/biasconv1d_23/kernelconv1d_23/biaslatent_vector/kernellatent_vector/biasdense_33/kerneldense_33/biasdense_34/kerneldense_34/biasdense_35/kerneldense_35/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/conv1d_22/kernel/mAdam/conv1d_22/bias/mAdam/conv1d_23/kernel/mAdam/conv1d_23/bias/mAdam/latent_vector/kernel/mAdam/latent_vector/bias/mAdam/dense_33/kernel/mAdam/dense_33/bias/mAdam/dense_34/kernel/mAdam/dense_34/bias/mAdam/dense_35/kernel/mAdam/dense_35/bias/mAdam/conv1d_22/kernel/vAdam/conv1d_22/bias/vAdam/conv1d_23/kernel/vAdam/conv1d_23/bias/vAdam/latent_vector/kernel/vAdam/latent_vector/bias/vAdam/dense_33/kernel/vAdam/dense_33/bias/vAdam/dense_34/kernel/vAdam/dense_34/bias/vAdam/dense_35/kernel/vAdam/dense_35/bias/v*7
Tin0
.2,*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference__traced_restore_863584ù

Ð	
û
I__inference_latent_vector_layer_call_and_return_conditional_losses_863216

inputs1
matmul_readvariableop_resource:	¨@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	¨@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ¨: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
 
_user_specified_nameinputs
¨7
Ý
C__inference_encoder_layer_call_and_return_conditional_losses_863017

inputsK
5conv1d_22_conv1d_expanddims_1_readvariableop_resource:7
)conv1d_22_biasadd_readvariableop_resource:K
5conv1d_23_conv1d_expanddims_1_readvariableop_resource:7
)conv1d_23_biasadd_readvariableop_resource:?
,latent_vector_matmul_readvariableop_resource:	¨@;
-latent_vector_biasadd_readvariableop_resource:@
identity¢ conv1d_22/BiasAdd/ReadVariableOp¢,conv1d_22/Conv1D/ExpandDims_1/ReadVariableOp¢ conv1d_23/BiasAdd/ReadVariableOp¢,conv1d_23/Conv1D/ExpandDims_1/ReadVariableOp¢$latent_vector/BiasAdd/ReadVariableOp¢#latent_vector/MatMul/ReadVariableOpJ
random_noise_4/ShapeShapeinputs*
T0*
_output_shapes
:f
!random_noise_4/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    h
#random_noise_4/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>Ê
1random_noise_4/random_normal/RandomStandardNormalRandomStandardNormalrandom_noise_4/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
dtype0*
seed±ÿå)*
seed2Ð´åÇ
 random_noise_4/random_normal/mulMul:random_noise_4/random_normal/RandomStandardNormal:output:0,random_noise_4/random_normal/stddev:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2­
random_noise_4/random_normalAddV2$random_noise_4/random_normal/mul:z:0*random_noise_4/random_normal/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2{
random_noise_4/addAddV2inputs random_noise_4/random_normal:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2i
random_noise_4/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          q
random_noise_4/MaxMaxrandom_noise_4/add:z:0random_noise_4/Const:output:0*
T0*
_output_shapes
: 
random_noise_4/truedivRealDivrandom_noise_4/add:z:0random_noise_4/Max:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2j
conv1d_22/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ©
conv1d_22/Conv1D/ExpandDims
ExpandDimsrandom_noise_4/truediv:z:0(conv1d_22/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2¦
,conv1d_22/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_22_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0c
!conv1d_22/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ¾
conv1d_22/Conv1D/ExpandDims_1
ExpandDims4conv1d_22/Conv1D/ExpandDims_1/ReadVariableOp:value:0*conv1d_22/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:Ë
conv1d_22/Conv1DConv2D$conv1d_22/Conv1D/ExpandDims:output:0&conv1d_22/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.*
paddingVALID*
strides

conv1d_22/Conv1D/SqueezeSqueezeconv1d_22/Conv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.*
squeeze_dims

ýÿÿÿÿÿÿÿÿ
 conv1d_22/BiasAdd/ReadVariableOpReadVariableOp)conv1d_22_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv1d_22/BiasAddBiasAdd!conv1d_22/Conv1D/Squeeze:output:0(conv1d_22/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.h
conv1d_22/ReluReluconv1d_22/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.j
conv1d_23/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ«
conv1d_23/Conv1D/ExpandDims
ExpandDimsconv1d_22/Relu:activations:0(conv1d_23/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.¦
,conv1d_23/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_23_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0c
!conv1d_23/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ¾
conv1d_23/Conv1D/ExpandDims_1
ExpandDims4conv1d_23/Conv1D/ExpandDims_1/ReadVariableOp:value:0*conv1d_23/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:Ë
conv1d_23/Conv1DConv2D$conv1d_23/Conv1D/ExpandDims:output:0&conv1d_23/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
paddingVALID*
strides

conv1d_23/Conv1D/SqueezeSqueezeconv1d_23/Conv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
squeeze_dims

ýÿÿÿÿÿÿÿÿ
 conv1d_23/BiasAdd/ReadVariableOpReadVariableOp)conv1d_23_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv1d_23/BiasAddBiasAdd!conv1d_23/Conv1D/Squeeze:output:0(conv1d_23/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*h
conv1d_23/ReluReluconv1d_23/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*a
flatten_11/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ¨   
flatten_11/ReshapeReshapeconv1d_23/Relu:activations:0flatten_11/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
#latent_vector/MatMul/ReadVariableOpReadVariableOp,latent_vector_matmul_readvariableop_resource*
_output_shapes
:	¨@*
dtype0
latent_vector/MatMulMatMulflatten_11/Reshape:output:0+latent_vector/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
$latent_vector/BiasAdd/ReadVariableOpReadVariableOp-latent_vector_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0 
latent_vector/BiasAddBiasAddlatent_vector/MatMul:product:0,latent_vector/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@m
IdentityIdentitylatent_vector/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@·
NoOpNoOp!^conv1d_22/BiasAdd/ReadVariableOp-^conv1d_22/Conv1D/ExpandDims_1/ReadVariableOp!^conv1d_23/BiasAdd/ReadVariableOp-^conv1d_23/Conv1D/ExpandDims_1/ReadVariableOp%^latent_vector/BiasAdd/ReadVariableOp$^latent_vector/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ2: : : : : : 2D
 conv1d_22/BiasAdd/ReadVariableOp conv1d_22/BiasAdd/ReadVariableOp2\
,conv1d_22/Conv1D/ExpandDims_1/ReadVariableOp,conv1d_22/Conv1D/ExpandDims_1/ReadVariableOp2D
 conv1d_23/BiasAdd/ReadVariableOp conv1d_23/BiasAdd/ReadVariableOp2\
,conv1d_23/Conv1D/ExpandDims_1/ReadVariableOp,conv1d_23/Conv1D/ExpandDims_1/ReadVariableOp2L
$latent_vector/BiasAdd/ReadVariableOp$latent_vector/BiasAdd/ReadVariableOp2J
#latent_vector/MatMul/ReadVariableOp#latent_vector/MatMul/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
 
_user_specified_nameinputs
ö$

C__inference_decoder_layer_call_and_return_conditional_losses_863084

inputs9
'dense_33_matmul_readvariableop_resource:@ 6
(dense_33_biasadd_readvariableop_resource: 9
'dense_34_matmul_readvariableop_resource:  6
(dense_34_biasadd_readvariableop_resource: 9
'dense_35_matmul_readvariableop_resource: 26
(dense_35_biasadd_readvariableop_resource:2
identity¢dense_33/BiasAdd/ReadVariableOp¢dense_33/MatMul/ReadVariableOp¢dense_34/BiasAdd/ReadVariableOp¢dense_34/MatMul/ReadVariableOp¢dense_35/BiasAdd/ReadVariableOp¢dense_35/MatMul/ReadVariableOp
dense_33/MatMul/ReadVariableOpReadVariableOp'dense_33_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype0{
dense_33/MatMulMatMulinputs&dense_33/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
dense_33/BiasAdd/ReadVariableOpReadVariableOp(dense_33_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
dense_33/BiasAddBiasAdddense_33/MatMul:product:0'dense_33/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ b
dense_33/ReluReludense_33/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
dense_34/MatMul/ReadVariableOpReadVariableOp'dense_34_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0
dense_34/MatMulMatMuldense_33/Relu:activations:0&dense_34/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
dense_34/BiasAdd/ReadVariableOpReadVariableOp(dense_34_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
dense_34/BiasAddBiasAdddense_34/MatMul:product:0'dense_34/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ b
dense_34/ReluReludense_34/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
dense_35/MatMul/ReadVariableOpReadVariableOp'dense_35_matmul_readvariableop_resource*
_output_shapes

: 2*
dtype0
dense_35/MatMulMatMuldense_34/Relu:activations:0&dense_35/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_35/BiasAdd/ReadVariableOpReadVariableOp(dense_35_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0
dense_35/BiasAddBiasAdddense_35/MatMul:product:0'dense_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2Y
reshape_11/ShapeShapedense_35/BiasAdd:output:0*
T0*
_output_shapes
:h
reshape_11/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: j
 reshape_11/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j
 reshape_11/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
reshape_11/strided_sliceStridedSlicereshape_11/Shape:output:0'reshape_11/strided_slice/stack:output:0)reshape_11/strided_slice/stack_1:output:0)reshape_11/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
reshape_11/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2\
reshape_11/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :»
reshape_11/Reshape/shapePack!reshape_11/strided_slice:output:0#reshape_11/Reshape/shape/1:output:0#reshape_11/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:
reshape_11/ReshapeReshapedense_35/BiasAdd:output:0!reshape_11/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2n
IdentityIdentityreshape_11/Reshape:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
NoOpNoOp ^dense_33/BiasAdd/ReadVariableOp^dense_33/MatMul/ReadVariableOp ^dense_34/BiasAdd/ReadVariableOp^dense_34/MatMul/ReadVariableOp ^dense_35/BiasAdd/ReadVariableOp^dense_35/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ@: : : : : : 2B
dense_33/BiasAdd/ReadVariableOpdense_33/BiasAdd/ReadVariableOp2@
dense_33/MatMul/ReadVariableOpdense_33/MatMul/ReadVariableOp2B
dense_34/BiasAdd/ReadVariableOpdense_34/BiasAdd/ReadVariableOp2@
dense_34/MatMul/ReadVariableOpdense_34/MatMul/ReadVariableOp2B
dense_35/BiasAdd/ReadVariableOpdense_35/BiasAdd/ReadVariableOp2@
dense_35/MatMul/ReadVariableOpdense_35/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
	

(__inference_decoder_layer_call_fn_862368
decoder_input
unknown:@ 
	unknown_0: 
	unknown_1:  
	unknown_2: 
	unknown_3: 2
	unknown_4:2
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCalldecoder_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_decoder_layer_call_and_return_conditional_losses_862336s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ@: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
'
_user_specified_namedecoder_input
ù

(__inference_encoder_layer_call_fn_862925

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:	¨@
	unknown_4:@
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_encoder_layer_call_and_return_conditional_losses_862099o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ2: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
 
_user_specified_nameinputs

±
,__inference_autoencoder_layer_call_fn_862741

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:	¨@
	unknown_4:@
	unknown_5:@ 
	unknown_6: 
	unknown_7:  
	unknown_8: 
	unknown_9: 2

unknown_10:2
identity¢StatefulPartitionedCallã
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_autoencoder_layer_call_and_return_conditional_losses_862530s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ2: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
 
_user_specified_nameinputs

¸
,__inference_autoencoder_layer_call_fn_862586
encoder_input
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:	¨@
	unknown_4:@
	unknown_5:@ 
	unknown_6: 
	unknown_7:  
	unknown_8: 
	unknown_9: 2

unknown_10:2
identity¢StatefulPartitionedCallê
StatefulPartitionedCallStatefulPartitionedCallencoder_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_autoencoder_layer_call_and_return_conditional_losses_862530s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ2: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
'
_user_specified_nameencoder_input
ã

C__inference_decoder_layer_call_and_return_conditional_losses_862246

inputs!
dense_33_862192:@ 
dense_33_862194: !
dense_34_862209:  
dense_34_862211: !
dense_35_862225: 2
dense_35_862227:2
identity¢ dense_33/StatefulPartitionedCall¢ dense_34/StatefulPartitionedCall¢ dense_35/StatefulPartitionedCallð
 dense_33/StatefulPartitionedCallStatefulPartitionedCallinputsdense_33_862192dense_33_862194*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_33_layer_call_and_return_conditional_losses_862191
 dense_34/StatefulPartitionedCallStatefulPartitionedCall)dense_33/StatefulPartitionedCall:output:0dense_34_862209dense_34_862211*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_34_layer_call_and_return_conditional_losses_862208
 dense_35/StatefulPartitionedCallStatefulPartitionedCall)dense_34/StatefulPartitionedCall:output:0dense_35_862225dense_35_862227*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_35_layer_call_and_return_conditional_losses_862224ã
reshape_11/PartitionedCallPartitionedCall)dense_35/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_reshape_11_layer_call_and_return_conditional_losses_862243v
IdentityIdentity#reshape_11/PartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2¯
NoOpNoOp!^dense_33/StatefulPartitionedCall!^dense_34/StatefulPartitionedCall!^dense_35/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ@: : : : : : 2D
 dense_33/StatefulPartitionedCall dense_33/StatefulPartitionedCall2D
 dense_34/StatefulPartitionedCall dense_34/StatefulPartitionedCall2D
 dense_35/StatefulPartitionedCall dense_35/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs


õ
D__inference_dense_33_layer_call_and_return_conditional_losses_862191

inputs0
matmul_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Â

)__inference_dense_35_layer_call_fn_863265

inputs
unknown: 2
	unknown_0:2
identity¢StatefulPartitionedCallÙ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_35_layer_call_and_return_conditional_losses_862224o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ó
º
G__inference_autoencoder_layer_call_and_return_conditional_losses_862646
encoder_input$
encoder_862619:
encoder_862621:$
encoder_862623:
encoder_862625:!
encoder_862627:	¨@
encoder_862629:@ 
decoder_862632:@ 
decoder_862634:  
decoder_862636:  
decoder_862638:  
decoder_862640: 2
decoder_862642:2
identity¢decoder/StatefulPartitionedCall¢encoder/StatefulPartitionedCall»
encoder/StatefulPartitionedCallStatefulPartitionedCallencoder_inputencoder_862619encoder_862621encoder_862623encoder_862625encoder_862627encoder_862629*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_encoder_layer_call_and_return_conditional_losses_862099Ú
decoder/StatefulPartitionedCallStatefulPartitionedCall(encoder/StatefulPartitionedCall:output:0decoder_862632decoder_862634decoder_862636decoder_862638decoder_862640decoder_862642*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_decoder_layer_call_and_return_conditional_losses_862336{
IdentityIdentity(decoder/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
NoOpNoOp ^decoder/StatefulPartitionedCall ^encoder/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ2: : : : : : : : : : : : 2B
decoder/StatefulPartitionedCalldecoder/StatefulPartitionedCall2B
encoder/StatefulPartitionedCallencoder/StatefulPartitionedCall:Z V
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
'
_user_specified_nameencoder_input
	

(__inference_encoder_layer_call_fn_862131
encoder_input
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:	¨@
	unknown_4:@
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallencoder_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_encoder_layer_call_and_return_conditional_losses_862099o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ2: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
'
_user_specified_nameencoder_input
Ö¨
Å
"__inference__traced_restore_863584
file_prefix7
!assignvariableop_conv1d_22_kernel:/
!assignvariableop_1_conv1d_22_bias:9
#assignvariableop_2_conv1d_23_kernel:/
!assignvariableop_3_conv1d_23_bias::
'assignvariableop_4_latent_vector_kernel:	¨@3
%assignvariableop_5_latent_vector_bias:@4
"assignvariableop_6_dense_33_kernel:@ .
 assignvariableop_7_dense_33_bias: 4
"assignvariableop_8_dense_34_kernel:  .
 assignvariableop_9_dense_34_bias: 5
#assignvariableop_10_dense_35_kernel: 2/
!assignvariableop_11_dense_35_bias:2'
assignvariableop_12_adam_iter:	 )
assignvariableop_13_adam_beta_1: )
assignvariableop_14_adam_beta_2: (
assignvariableop_15_adam_decay: 0
&assignvariableop_16_adam_learning_rate: #
assignvariableop_17_total: #
assignvariableop_18_count: A
+assignvariableop_19_adam_conv1d_22_kernel_m:7
)assignvariableop_20_adam_conv1d_22_bias_m:A
+assignvariableop_21_adam_conv1d_23_kernel_m:7
)assignvariableop_22_adam_conv1d_23_bias_m:B
/assignvariableop_23_adam_latent_vector_kernel_m:	¨@;
-assignvariableop_24_adam_latent_vector_bias_m:@<
*assignvariableop_25_adam_dense_33_kernel_m:@ 6
(assignvariableop_26_adam_dense_33_bias_m: <
*assignvariableop_27_adam_dense_34_kernel_m:  6
(assignvariableop_28_adam_dense_34_bias_m: <
*assignvariableop_29_adam_dense_35_kernel_m: 26
(assignvariableop_30_adam_dense_35_bias_m:2A
+assignvariableop_31_adam_conv1d_22_kernel_v:7
)assignvariableop_32_adam_conv1d_22_bias_v:A
+assignvariableop_33_adam_conv1d_23_kernel_v:7
)assignvariableop_34_adam_conv1d_23_bias_v:B
/assignvariableop_35_adam_latent_vector_kernel_v:	¨@;
-assignvariableop_36_adam_latent_vector_bias_v:@<
*assignvariableop_37_adam_dense_33_kernel_v:@ 6
(assignvariableop_38_adam_dense_33_bias_v: <
*assignvariableop_39_adam_dense_34_kernel_v:  6
(assignvariableop_40_adam_dense_34_bias_v: <
*assignvariableop_41_adam_dense_35_kernel_v: 26
(assignvariableop_42_adam_dense_35_bias_v:2
identity_44¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9¤
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:,*
dtype0*Ê
valueÀB½,B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHÈ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:,*
dtype0*k
valuebB`,B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ý
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Æ
_output_shapes³
°::::::::::::::::::::::::::::::::::::::::::::*:
dtypes0
.2,	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp!assignvariableop_conv1d_22_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv1d_22_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp#assignvariableop_2_conv1d_23_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp!assignvariableop_3_conv1d_23_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp'assignvariableop_4_latent_vector_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp%assignvariableop_5_latent_vector_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp"assignvariableop_6_dense_33_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOp assignvariableop_7_dense_33_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOp"assignvariableop_8_dense_34_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOp assignvariableop_9_dense_34_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOp#assignvariableop_10_dense_35_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOp!assignvariableop_11_dense_35_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_iterIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOpassignvariableop_13_adam_beta_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOpassignvariableop_14_adam_beta_2Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOpassignvariableop_15_adam_decayIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOp&assignvariableop_16_adam_learning_rateIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOpassignvariableop_17_totalIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOpassignvariableop_18_countIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOp+assignvariableop_19_adam_conv1d_22_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_20AssignVariableOp)assignvariableop_20_adam_conv1d_22_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOp+assignvariableop_21_adam_conv1d_23_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_22AssignVariableOp)assignvariableop_22_adam_conv1d_23_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_23AssignVariableOp/assignvariableop_23_adam_latent_vector_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_24AssignVariableOp-assignvariableop_24_adam_latent_vector_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_25AssignVariableOp*assignvariableop_25_adam_dense_33_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_26AssignVariableOp(assignvariableop_26_adam_dense_33_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_27AssignVariableOp*assignvariableop_27_adam_dense_34_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_28AssignVariableOp(assignvariableop_28_adam_dense_34_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_29AssignVariableOp*assignvariableop_29_adam_dense_35_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_30AssignVariableOp(assignvariableop_30_adam_dense_35_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_31AssignVariableOp+assignvariableop_31_adam_conv1d_22_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_32AssignVariableOp)assignvariableop_32_adam_conv1d_22_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_33AssignVariableOp+assignvariableop_33_adam_conv1d_23_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_34AssignVariableOp)assignvariableop_34_adam_conv1d_23_bias_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_35AssignVariableOp/assignvariableop_35_adam_latent_vector_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_36AssignVariableOp-assignvariableop_36_adam_latent_vector_bias_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_37AssignVariableOp*assignvariableop_37_adam_dense_33_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_38AssignVariableOp(assignvariableop_38_adam_dense_33_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_39AssignVariableOp*assignvariableop_39_adam_dense_34_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_40AssignVariableOp(assignvariableop_40_adam_dense_34_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_41AssignVariableOp*assignvariableop_41_adam_dense_35_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_42AssignVariableOp(assignvariableop_42_adam_dense_35_bias_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 
Identity_43Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_44IdentityIdentity_43:output:0^NoOp_1*
T0*
_output_shapes
: î
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_44Identity_44:output:0*k
_input_shapesZ
X: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
7
Ý
C__inference_encoder_layer_call_and_return_conditional_losses_862971

inputsK
5conv1d_22_conv1d_expanddims_1_readvariableop_resource:7
)conv1d_22_biasadd_readvariableop_resource:K
5conv1d_23_conv1d_expanddims_1_readvariableop_resource:7
)conv1d_23_biasadd_readvariableop_resource:?
,latent_vector_matmul_readvariableop_resource:	¨@;
-latent_vector_biasadd_readvariableop_resource:@
identity¢ conv1d_22/BiasAdd/ReadVariableOp¢,conv1d_22/Conv1D/ExpandDims_1/ReadVariableOp¢ conv1d_23/BiasAdd/ReadVariableOp¢,conv1d_23/Conv1D/ExpandDims_1/ReadVariableOp¢$latent_vector/BiasAdd/ReadVariableOp¢#latent_vector/MatMul/ReadVariableOpJ
random_noise_4/ShapeShapeinputs*
T0*
_output_shapes
:f
!random_noise_4/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    h
#random_noise_4/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>Ê
1random_noise_4/random_normal/RandomStandardNormalRandomStandardNormalrandom_noise_4/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
dtype0*
seed±ÿå)*
seed2·ÌÇ
 random_noise_4/random_normal/mulMul:random_noise_4/random_normal/RandomStandardNormal:output:0,random_noise_4/random_normal/stddev:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2­
random_noise_4/random_normalAddV2$random_noise_4/random_normal/mul:z:0*random_noise_4/random_normal/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2{
random_noise_4/addAddV2inputs random_noise_4/random_normal:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2i
random_noise_4/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          a
random_noise_4/MaxMaxinputsrandom_noise_4/Const:output:0*
T0*
_output_shapes
: |
random_noise_4/truedivRealDivinputsrandom_noise_4/Max:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2j
conv1d_22/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ©
conv1d_22/Conv1D/ExpandDims
ExpandDimsrandom_noise_4/truediv:z:0(conv1d_22/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2¦
,conv1d_22/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_22_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0c
!conv1d_22/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ¾
conv1d_22/Conv1D/ExpandDims_1
ExpandDims4conv1d_22/Conv1D/ExpandDims_1/ReadVariableOp:value:0*conv1d_22/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:Ë
conv1d_22/Conv1DConv2D$conv1d_22/Conv1D/ExpandDims:output:0&conv1d_22/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.*
paddingVALID*
strides

conv1d_22/Conv1D/SqueezeSqueezeconv1d_22/Conv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.*
squeeze_dims

ýÿÿÿÿÿÿÿÿ
 conv1d_22/BiasAdd/ReadVariableOpReadVariableOp)conv1d_22_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv1d_22/BiasAddBiasAdd!conv1d_22/Conv1D/Squeeze:output:0(conv1d_22/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.h
conv1d_22/ReluReluconv1d_22/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.j
conv1d_23/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ«
conv1d_23/Conv1D/ExpandDims
ExpandDimsconv1d_22/Relu:activations:0(conv1d_23/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.¦
,conv1d_23/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_23_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0c
!conv1d_23/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ¾
conv1d_23/Conv1D/ExpandDims_1
ExpandDims4conv1d_23/Conv1D/ExpandDims_1/ReadVariableOp:value:0*conv1d_23/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:Ë
conv1d_23/Conv1DConv2D$conv1d_23/Conv1D/ExpandDims:output:0&conv1d_23/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
paddingVALID*
strides

conv1d_23/Conv1D/SqueezeSqueezeconv1d_23/Conv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
squeeze_dims

ýÿÿÿÿÿÿÿÿ
 conv1d_23/BiasAdd/ReadVariableOpReadVariableOp)conv1d_23_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv1d_23/BiasAddBiasAdd!conv1d_23/Conv1D/Squeeze:output:0(conv1d_23/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*h
conv1d_23/ReluReluconv1d_23/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*a
flatten_11/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ¨   
flatten_11/ReshapeReshapeconv1d_23/Relu:activations:0flatten_11/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
#latent_vector/MatMul/ReadVariableOpReadVariableOp,latent_vector_matmul_readvariableop_resource*
_output_shapes
:	¨@*
dtype0
latent_vector/MatMulMatMulflatten_11/Reshape:output:0+latent_vector/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
$latent_vector/BiasAdd/ReadVariableOpReadVariableOp-latent_vector_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0 
latent_vector/BiasAddBiasAddlatent_vector/MatMul:product:0,latent_vector/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@m
IdentityIdentitylatent_vector/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@·
NoOpNoOp!^conv1d_22/BiasAdd/ReadVariableOp-^conv1d_22/Conv1D/ExpandDims_1/ReadVariableOp!^conv1d_23/BiasAdd/ReadVariableOp-^conv1d_23/Conv1D/ExpandDims_1/ReadVariableOp%^latent_vector/BiasAdd/ReadVariableOp$^latent_vector/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ2: : : : : : 2D
 conv1d_22/BiasAdd/ReadVariableOp conv1d_22/BiasAdd/ReadVariableOp2\
,conv1d_22/Conv1D/ExpandDims_1/ReadVariableOp,conv1d_22/Conv1D/ExpandDims_1/ReadVariableOp2D
 conv1d_23/BiasAdd/ReadVariableOp conv1d_23/BiasAdd/ReadVariableOp2\
,conv1d_23/Conv1D/ExpandDims_1/ReadVariableOp,conv1d_23/Conv1D/ExpandDims_1/ReadVariableOp2L
$latent_vector/BiasAdd/ReadVariableOp$latent_vector/BiasAdd/ReadVariableOp2J
#latent_vector/MatMul/ReadVariableOp#latent_vector/MatMul/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
 
_user_specified_nameinputs
À
b
F__inference_flatten_11_layer_call_and_return_conditional_losses_863197

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ¨   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ*:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
 
_user_specified_nameinputs


õ
D__inference_dense_34_layer_call_and_return_conditional_losses_862208

inputs0
matmul_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ã

C__inference_decoder_layer_call_and_return_conditional_losses_862336

inputs!
dense_33_862319:@ 
dense_33_862321: !
dense_34_862324:  
dense_34_862326: !
dense_35_862329: 2
dense_35_862331:2
identity¢ dense_33/StatefulPartitionedCall¢ dense_34/StatefulPartitionedCall¢ dense_35/StatefulPartitionedCallð
 dense_33/StatefulPartitionedCallStatefulPartitionedCallinputsdense_33_862319dense_33_862321*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_33_layer_call_and_return_conditional_losses_862191
 dense_34/StatefulPartitionedCallStatefulPartitionedCall)dense_33/StatefulPartitionedCall:output:0dense_34_862324dense_34_862326*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_34_layer_call_and_return_conditional_losses_862208
 dense_35/StatefulPartitionedCallStatefulPartitionedCall)dense_34/StatefulPartitionedCall:output:0dense_35_862329dense_35_862331*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_35_layer_call_and_return_conditional_losses_862224ã
reshape_11/PartitionedCallPartitionedCall)dense_35/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_reshape_11_layer_call_and_return_conditional_losses_862243v
IdentityIdentity#reshape_11/PartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2¯
NoOpNoOp!^dense_33/StatefulPartitionedCall!^dense_34/StatefulPartitionedCall!^dense_35/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ@: : : : : : 2D
 dense_33/StatefulPartitionedCall dense_33/StatefulPartitionedCall2D
 dense_34/StatefulPartitionedCall dense_34/StatefulPartitionedCall2D
 dense_35/StatefulPartitionedCall dense_35/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ø

b
F__inference_reshape_11_layer_call_and_return_conditional_losses_862243

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2Q
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:h
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2\
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ2:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
 
_user_specified_nameinputs

¸
,__inference_autoencoder_layer_call_fn_862469
encoder_input
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:	¨@
	unknown_4:@
	unknown_5:@ 
	unknown_6: 
	unknown_7:  
	unknown_8: 
	unknown_9: 2

unknown_10:2
identity¢StatefulPartitionedCallê
StatefulPartitionedCallStatefulPartitionedCallencoder_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_autoencoder_layer_call_and_return_conditional_losses_862442s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ2: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
'
_user_specified_nameencoder_input
­d
Ã
G__inference_autoencoder_layer_call_and_return_conditional_losses_862816

inputsS
=encoder_conv1d_22_conv1d_expanddims_1_readvariableop_resource:?
1encoder_conv1d_22_biasadd_readvariableop_resource:S
=encoder_conv1d_23_conv1d_expanddims_1_readvariableop_resource:?
1encoder_conv1d_23_biasadd_readvariableop_resource:G
4encoder_latent_vector_matmul_readvariableop_resource:	¨@C
5encoder_latent_vector_biasadd_readvariableop_resource:@A
/decoder_dense_33_matmul_readvariableop_resource:@ >
0decoder_dense_33_biasadd_readvariableop_resource: A
/decoder_dense_34_matmul_readvariableop_resource:  >
0decoder_dense_34_biasadd_readvariableop_resource: A
/decoder_dense_35_matmul_readvariableop_resource: 2>
0decoder_dense_35_biasadd_readvariableop_resource:2
identity¢'decoder/dense_33/BiasAdd/ReadVariableOp¢&decoder/dense_33/MatMul/ReadVariableOp¢'decoder/dense_34/BiasAdd/ReadVariableOp¢&decoder/dense_34/MatMul/ReadVariableOp¢'decoder/dense_35/BiasAdd/ReadVariableOp¢&decoder/dense_35/MatMul/ReadVariableOp¢(encoder/conv1d_22/BiasAdd/ReadVariableOp¢4encoder/conv1d_22/Conv1D/ExpandDims_1/ReadVariableOp¢(encoder/conv1d_23/BiasAdd/ReadVariableOp¢4encoder/conv1d_23/Conv1D/ExpandDims_1/ReadVariableOp¢,encoder/latent_vector/BiasAdd/ReadVariableOp¢+encoder/latent_vector/MatMul/ReadVariableOpR
encoder/random_noise_4/ShapeShapeinputs*
T0*
_output_shapes
:n
)encoder/random_noise_4/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    p
+encoder/random_noise_4/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>Ú
9encoder/random_noise_4/random_normal/RandomStandardNormalRandomStandardNormal%encoder/random_noise_4/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
dtype0*
seed±ÿå)*
seed2¾¿©ß
(encoder/random_noise_4/random_normal/mulMulBencoder/random_noise_4/random_normal/RandomStandardNormal:output:04encoder/random_noise_4/random_normal/stddev:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2Å
$encoder/random_noise_4/random_normalAddV2,encoder/random_noise_4/random_normal/mul:z:02encoder/random_noise_4/random_normal/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
encoder/random_noise_4/addAddV2inputs(encoder/random_noise_4/random_normal:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2q
encoder/random_noise_4/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          q
encoder/random_noise_4/MaxMaxinputs%encoder/random_noise_4/Const:output:0*
T0*
_output_shapes
: 
encoder/random_noise_4/truedivRealDivinputs#encoder/random_noise_4/Max:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2r
'encoder/conv1d_22/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿÁ
#encoder/conv1d_22/Conv1D/ExpandDims
ExpandDims"encoder/random_noise_4/truediv:z:00encoder/conv1d_22/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2¶
4encoder/conv1d_22/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp=encoder_conv1d_22_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0k
)encoder/conv1d_22/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : Ö
%encoder/conv1d_22/Conv1D/ExpandDims_1
ExpandDims<encoder/conv1d_22/Conv1D/ExpandDims_1/ReadVariableOp:value:02encoder/conv1d_22/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:ã
encoder/conv1d_22/Conv1DConv2D,encoder/conv1d_22/Conv1D/ExpandDims:output:0.encoder/conv1d_22/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.*
paddingVALID*
strides
¤
 encoder/conv1d_22/Conv1D/SqueezeSqueeze!encoder/conv1d_22/Conv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.*
squeeze_dims

ýÿÿÿÿÿÿÿÿ
(encoder/conv1d_22/BiasAdd/ReadVariableOpReadVariableOp1encoder_conv1d_22_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0·
encoder/conv1d_22/BiasAddBiasAdd)encoder/conv1d_22/Conv1D/Squeeze:output:00encoder/conv1d_22/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.x
encoder/conv1d_22/ReluRelu"encoder/conv1d_22/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.r
'encoder/conv1d_23/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿÃ
#encoder/conv1d_23/Conv1D/ExpandDims
ExpandDims$encoder/conv1d_22/Relu:activations:00encoder/conv1d_23/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.¶
4encoder/conv1d_23/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp=encoder_conv1d_23_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0k
)encoder/conv1d_23/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : Ö
%encoder/conv1d_23/Conv1D/ExpandDims_1
ExpandDims<encoder/conv1d_23/Conv1D/ExpandDims_1/ReadVariableOp:value:02encoder/conv1d_23/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:ã
encoder/conv1d_23/Conv1DConv2D,encoder/conv1d_23/Conv1D/ExpandDims:output:0.encoder/conv1d_23/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
paddingVALID*
strides
¤
 encoder/conv1d_23/Conv1D/SqueezeSqueeze!encoder/conv1d_23/Conv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
squeeze_dims

ýÿÿÿÿÿÿÿÿ
(encoder/conv1d_23/BiasAdd/ReadVariableOpReadVariableOp1encoder_conv1d_23_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0·
encoder/conv1d_23/BiasAddBiasAdd)encoder/conv1d_23/Conv1D/Squeeze:output:00encoder/conv1d_23/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*x
encoder/conv1d_23/ReluRelu"encoder/conv1d_23/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*i
encoder/flatten_11/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ¨   ¡
encoder/flatten_11/ReshapeReshape$encoder/conv1d_23/Relu:activations:0!encoder/flatten_11/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨¡
+encoder/latent_vector/MatMul/ReadVariableOpReadVariableOp4encoder_latent_vector_matmul_readvariableop_resource*
_output_shapes
:	¨@*
dtype0²
encoder/latent_vector/MatMulMatMul#encoder/flatten_11/Reshape:output:03encoder/latent_vector/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
,encoder/latent_vector/BiasAdd/ReadVariableOpReadVariableOp5encoder_latent_vector_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¸
encoder/latent_vector/BiasAddBiasAdd&encoder/latent_vector/MatMul:product:04encoder/latent_vector/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
&decoder/dense_33/MatMul/ReadVariableOpReadVariableOp/decoder_dense_33_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype0«
decoder/dense_33/MatMulMatMul&encoder/latent_vector/BiasAdd:output:0.decoder/dense_33/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
'decoder/dense_33/BiasAdd/ReadVariableOpReadVariableOp0decoder_dense_33_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0©
decoder/dense_33/BiasAddBiasAdd!decoder/dense_33/MatMul:product:0/decoder/dense_33/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ r
decoder/dense_33/ReluRelu!decoder/dense_33/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
&decoder/dense_34/MatMul/ReadVariableOpReadVariableOp/decoder_dense_34_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0¨
decoder/dense_34/MatMulMatMul#decoder/dense_33/Relu:activations:0.decoder/dense_34/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
'decoder/dense_34/BiasAdd/ReadVariableOpReadVariableOp0decoder_dense_34_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0©
decoder/dense_34/BiasAddBiasAdd!decoder/dense_34/MatMul:product:0/decoder/dense_34/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ r
decoder/dense_34/ReluRelu!decoder/dense_34/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
&decoder/dense_35/MatMul/ReadVariableOpReadVariableOp/decoder_dense_35_matmul_readvariableop_resource*
_output_shapes

: 2*
dtype0¨
decoder/dense_35/MatMulMatMul#decoder/dense_34/Relu:activations:0.decoder/dense_35/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
'decoder/dense_35/BiasAdd/ReadVariableOpReadVariableOp0decoder_dense_35_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0©
decoder/dense_35/BiasAddBiasAdd!decoder/dense_35/MatMul:product:0/decoder/dense_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2i
decoder/reshape_11/ShapeShape!decoder/dense_35/BiasAdd:output:0*
T0*
_output_shapes
:p
&decoder/reshape_11/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(decoder/reshape_11/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(decoder/reshape_11/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:°
 decoder/reshape_11/strided_sliceStridedSlice!decoder/reshape_11/Shape:output:0/decoder/reshape_11/strided_slice/stack:output:01decoder/reshape_11/strided_slice/stack_1:output:01decoder/reshape_11/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"decoder/reshape_11/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2d
"decoder/reshape_11/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Û
 decoder/reshape_11/Reshape/shapePack)decoder/reshape_11/strided_slice:output:0+decoder/reshape_11/Reshape/shape/1:output:0+decoder/reshape_11/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:©
decoder/reshape_11/ReshapeReshape!decoder/dense_35/BiasAdd:output:0)decoder/reshape_11/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2v
IdentityIdentity#decoder/reshape_11/Reshape:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2à
NoOpNoOp(^decoder/dense_33/BiasAdd/ReadVariableOp'^decoder/dense_33/MatMul/ReadVariableOp(^decoder/dense_34/BiasAdd/ReadVariableOp'^decoder/dense_34/MatMul/ReadVariableOp(^decoder/dense_35/BiasAdd/ReadVariableOp'^decoder/dense_35/MatMul/ReadVariableOp)^encoder/conv1d_22/BiasAdd/ReadVariableOp5^encoder/conv1d_22/Conv1D/ExpandDims_1/ReadVariableOp)^encoder/conv1d_23/BiasAdd/ReadVariableOp5^encoder/conv1d_23/Conv1D/ExpandDims_1/ReadVariableOp-^encoder/latent_vector/BiasAdd/ReadVariableOp,^encoder/latent_vector/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ2: : : : : : : : : : : : 2R
'decoder/dense_33/BiasAdd/ReadVariableOp'decoder/dense_33/BiasAdd/ReadVariableOp2P
&decoder/dense_33/MatMul/ReadVariableOp&decoder/dense_33/MatMul/ReadVariableOp2R
'decoder/dense_34/BiasAdd/ReadVariableOp'decoder/dense_34/BiasAdd/ReadVariableOp2P
&decoder/dense_34/MatMul/ReadVariableOp&decoder/dense_34/MatMul/ReadVariableOp2R
'decoder/dense_35/BiasAdd/ReadVariableOp'decoder/dense_35/BiasAdd/ReadVariableOp2P
&decoder/dense_35/MatMul/ReadVariableOp&decoder/dense_35/MatMul/ReadVariableOp2T
(encoder/conv1d_22/BiasAdd/ReadVariableOp(encoder/conv1d_22/BiasAdd/ReadVariableOp2l
4encoder/conv1d_22/Conv1D/ExpandDims_1/ReadVariableOp4encoder/conv1d_22/Conv1D/ExpandDims_1/ReadVariableOp2T
(encoder/conv1d_23/BiasAdd/ReadVariableOp(encoder/conv1d_23/BiasAdd/ReadVariableOp2l
4encoder/conv1d_23/Conv1D/ExpandDims_1/ReadVariableOp4encoder/conv1d_23/Conv1D/ExpandDims_1/ReadVariableOp2\
,encoder/latent_vector/BiasAdd/ReadVariableOp,encoder/latent_vector/BiasAdd/ReadVariableOp2Z
+encoder/latent_vector/MatMul/ReadVariableOp+encoder/latent_vector/MatMul/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
 
_user_specified_nameinputs
¢v
Ä
!__inference__wrapped_model_861910
encoder_input_
Iautoencoder_encoder_conv1d_22_conv1d_expanddims_1_readvariableop_resource:K
=autoencoder_encoder_conv1d_22_biasadd_readvariableop_resource:_
Iautoencoder_encoder_conv1d_23_conv1d_expanddims_1_readvariableop_resource:K
=autoencoder_encoder_conv1d_23_biasadd_readvariableop_resource:S
@autoencoder_encoder_latent_vector_matmul_readvariableop_resource:	¨@O
Aautoencoder_encoder_latent_vector_biasadd_readvariableop_resource:@M
;autoencoder_decoder_dense_33_matmul_readvariableop_resource:@ J
<autoencoder_decoder_dense_33_biasadd_readvariableop_resource: M
;autoencoder_decoder_dense_34_matmul_readvariableop_resource:  J
<autoencoder_decoder_dense_34_biasadd_readvariableop_resource: M
;autoencoder_decoder_dense_35_matmul_readvariableop_resource: 2J
<autoencoder_decoder_dense_35_biasadd_readvariableop_resource:2
identity¢3autoencoder/decoder/dense_33/BiasAdd/ReadVariableOp¢2autoencoder/decoder/dense_33/MatMul/ReadVariableOp¢3autoencoder/decoder/dense_34/BiasAdd/ReadVariableOp¢2autoencoder/decoder/dense_34/MatMul/ReadVariableOp¢3autoencoder/decoder/dense_35/BiasAdd/ReadVariableOp¢2autoencoder/decoder/dense_35/MatMul/ReadVariableOp¢4autoencoder/encoder/conv1d_22/BiasAdd/ReadVariableOp¢@autoencoder/encoder/conv1d_22/Conv1D/ExpandDims_1/ReadVariableOp¢4autoencoder/encoder/conv1d_23/BiasAdd/ReadVariableOp¢@autoencoder/encoder/conv1d_23/Conv1D/ExpandDims_1/ReadVariableOp¢8autoencoder/encoder/latent_vector/BiasAdd/ReadVariableOp¢7autoencoder/encoder/latent_vector/MatMul/ReadVariableOpe
(autoencoder/encoder/random_noise_4/ShapeShapeencoder_input*
T0*
_output_shapes
:z
5autoencoder/encoder/random_noise_4/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    |
7autoencoder/encoder/random_noise_4/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>ò
Eautoencoder/encoder/random_noise_4/random_normal/RandomStandardNormalRandomStandardNormal1autoencoder/encoder/random_noise_4/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
dtype0*
seed±ÿå)*
seed2³Ý´
4autoencoder/encoder/random_noise_4/random_normal/mulMulNautoencoder/encoder/random_noise_4/random_normal/RandomStandardNormal:output:0@autoencoder/encoder/random_noise_4/random_normal/stddev:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2é
0autoencoder/encoder/random_noise_4/random_normalAddV28autoencoder/encoder/random_noise_4/random_normal/mul:z:0>autoencoder/encoder/random_noise_4/random_normal/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2ª
&autoencoder/encoder/random_noise_4/addAddV2encoder_input4autoencoder/encoder/random_noise_4/random_normal:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2}
(autoencoder/encoder/random_noise_4/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          
&autoencoder/encoder/random_noise_4/MaxMaxencoder_input1autoencoder/encoder/random_noise_4/Const:output:0*
T0*
_output_shapes
: «
*autoencoder/encoder/random_noise_4/truedivRealDivencoder_input/autoencoder/encoder/random_noise_4/Max:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2~
3autoencoder/encoder/conv1d_22/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿå
/autoencoder/encoder/conv1d_22/Conv1D/ExpandDims
ExpandDims.autoencoder/encoder/random_noise_4/truediv:z:0<autoencoder/encoder/conv1d_22/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2Î
@autoencoder/encoder/conv1d_22/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpIautoencoder_encoder_conv1d_22_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0w
5autoencoder/encoder/conv1d_22/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ú
1autoencoder/encoder/conv1d_22/Conv1D/ExpandDims_1
ExpandDimsHautoencoder/encoder/conv1d_22/Conv1D/ExpandDims_1/ReadVariableOp:value:0>autoencoder/encoder/conv1d_22/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:
$autoencoder/encoder/conv1d_22/Conv1DConv2D8autoencoder/encoder/conv1d_22/Conv1D/ExpandDims:output:0:autoencoder/encoder/conv1d_22/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.*
paddingVALID*
strides
¼
,autoencoder/encoder/conv1d_22/Conv1D/SqueezeSqueeze-autoencoder/encoder/conv1d_22/Conv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.*
squeeze_dims

ýÿÿÿÿÿÿÿÿ®
4autoencoder/encoder/conv1d_22/BiasAdd/ReadVariableOpReadVariableOp=autoencoder_encoder_conv1d_22_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Û
%autoencoder/encoder/conv1d_22/BiasAddBiasAdd5autoencoder/encoder/conv1d_22/Conv1D/Squeeze:output:0<autoencoder/encoder/conv1d_22/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.
"autoencoder/encoder/conv1d_22/ReluRelu.autoencoder/encoder/conv1d_22/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.~
3autoencoder/encoder/conv1d_23/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿç
/autoencoder/encoder/conv1d_23/Conv1D/ExpandDims
ExpandDims0autoencoder/encoder/conv1d_22/Relu:activations:0<autoencoder/encoder/conv1d_23/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.Î
@autoencoder/encoder/conv1d_23/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpIautoencoder_encoder_conv1d_23_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0w
5autoencoder/encoder/conv1d_23/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ú
1autoencoder/encoder/conv1d_23/Conv1D/ExpandDims_1
ExpandDimsHautoencoder/encoder/conv1d_23/Conv1D/ExpandDims_1/ReadVariableOp:value:0>autoencoder/encoder/conv1d_23/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:
$autoencoder/encoder/conv1d_23/Conv1DConv2D8autoencoder/encoder/conv1d_23/Conv1D/ExpandDims:output:0:autoencoder/encoder/conv1d_23/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
paddingVALID*
strides
¼
,autoencoder/encoder/conv1d_23/Conv1D/SqueezeSqueeze-autoencoder/encoder/conv1d_23/Conv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
squeeze_dims

ýÿÿÿÿÿÿÿÿ®
4autoencoder/encoder/conv1d_23/BiasAdd/ReadVariableOpReadVariableOp=autoencoder_encoder_conv1d_23_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Û
%autoencoder/encoder/conv1d_23/BiasAddBiasAdd5autoencoder/encoder/conv1d_23/Conv1D/Squeeze:output:0<autoencoder/encoder/conv1d_23/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
"autoencoder/encoder/conv1d_23/ReluRelu.autoencoder/encoder/conv1d_23/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*u
$autoencoder/encoder/flatten_11/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ¨   Å
&autoencoder/encoder/flatten_11/ReshapeReshape0autoencoder/encoder/conv1d_23/Relu:activations:0-autoencoder/encoder/flatten_11/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨¹
7autoencoder/encoder/latent_vector/MatMul/ReadVariableOpReadVariableOp@autoencoder_encoder_latent_vector_matmul_readvariableop_resource*
_output_shapes
:	¨@*
dtype0Ö
(autoencoder/encoder/latent_vector/MatMulMatMul/autoencoder/encoder/flatten_11/Reshape:output:0?autoencoder/encoder/latent_vector/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¶
8autoencoder/encoder/latent_vector/BiasAdd/ReadVariableOpReadVariableOpAautoencoder_encoder_latent_vector_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ü
)autoencoder/encoder/latent_vector/BiasAddBiasAdd2autoencoder/encoder/latent_vector/MatMul:product:0@autoencoder/encoder/latent_vector/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@®
2autoencoder/decoder/dense_33/MatMul/ReadVariableOpReadVariableOp;autoencoder_decoder_dense_33_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype0Ï
#autoencoder/decoder/dense_33/MatMulMatMul2autoencoder/encoder/latent_vector/BiasAdd:output:0:autoencoder/decoder/dense_33/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¬
3autoencoder/decoder/dense_33/BiasAdd/ReadVariableOpReadVariableOp<autoencoder_decoder_dense_33_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Í
$autoencoder/decoder/dense_33/BiasAddBiasAdd-autoencoder/decoder/dense_33/MatMul:product:0;autoencoder/decoder/dense_33/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
!autoencoder/decoder/dense_33/ReluRelu-autoencoder/decoder/dense_33/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ®
2autoencoder/decoder/dense_34/MatMul/ReadVariableOpReadVariableOp;autoencoder_decoder_dense_34_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0Ì
#autoencoder/decoder/dense_34/MatMulMatMul/autoencoder/decoder/dense_33/Relu:activations:0:autoencoder/decoder/dense_34/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¬
3autoencoder/decoder/dense_34/BiasAdd/ReadVariableOpReadVariableOp<autoencoder_decoder_dense_34_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Í
$autoencoder/decoder/dense_34/BiasAddBiasAdd-autoencoder/decoder/dense_34/MatMul:product:0;autoencoder/decoder/dense_34/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
!autoencoder/decoder/dense_34/ReluRelu-autoencoder/decoder/dense_34/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ®
2autoencoder/decoder/dense_35/MatMul/ReadVariableOpReadVariableOp;autoencoder_decoder_dense_35_matmul_readvariableop_resource*
_output_shapes

: 2*
dtype0Ì
#autoencoder/decoder/dense_35/MatMulMatMul/autoencoder/decoder/dense_34/Relu:activations:0:autoencoder/decoder/dense_35/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2¬
3autoencoder/decoder/dense_35/BiasAdd/ReadVariableOpReadVariableOp<autoencoder_decoder_dense_35_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0Í
$autoencoder/decoder/dense_35/BiasAddBiasAdd-autoencoder/decoder/dense_35/MatMul:product:0;autoencoder/decoder/dense_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
$autoencoder/decoder/reshape_11/ShapeShape-autoencoder/decoder/dense_35/BiasAdd:output:0*
T0*
_output_shapes
:|
2autoencoder/decoder/reshape_11/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ~
4autoencoder/decoder/reshape_11/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:~
4autoencoder/decoder/reshape_11/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ì
,autoencoder/decoder/reshape_11/strided_sliceStridedSlice-autoencoder/decoder/reshape_11/Shape:output:0;autoencoder/decoder/reshape_11/strided_slice/stack:output:0=autoencoder/decoder/reshape_11/strided_slice/stack_1:output:0=autoencoder/decoder/reshape_11/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskp
.autoencoder/decoder/reshape_11/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2p
.autoencoder/decoder/reshape_11/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
,autoencoder/decoder/reshape_11/Reshape/shapePack5autoencoder/decoder/reshape_11/strided_slice:output:07autoencoder/decoder/reshape_11/Reshape/shape/1:output:07autoencoder/decoder/reshape_11/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:Í
&autoencoder/decoder/reshape_11/ReshapeReshape-autoencoder/decoder/dense_35/BiasAdd:output:05autoencoder/decoder/reshape_11/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
IdentityIdentity/autoencoder/decoder/reshape_11/Reshape:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2ð
NoOpNoOp4^autoencoder/decoder/dense_33/BiasAdd/ReadVariableOp3^autoencoder/decoder/dense_33/MatMul/ReadVariableOp4^autoencoder/decoder/dense_34/BiasAdd/ReadVariableOp3^autoencoder/decoder/dense_34/MatMul/ReadVariableOp4^autoencoder/decoder/dense_35/BiasAdd/ReadVariableOp3^autoencoder/decoder/dense_35/MatMul/ReadVariableOp5^autoencoder/encoder/conv1d_22/BiasAdd/ReadVariableOpA^autoencoder/encoder/conv1d_22/Conv1D/ExpandDims_1/ReadVariableOp5^autoencoder/encoder/conv1d_23/BiasAdd/ReadVariableOpA^autoencoder/encoder/conv1d_23/Conv1D/ExpandDims_1/ReadVariableOp9^autoencoder/encoder/latent_vector/BiasAdd/ReadVariableOp8^autoencoder/encoder/latent_vector/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ2: : : : : : : : : : : : 2j
3autoencoder/decoder/dense_33/BiasAdd/ReadVariableOp3autoencoder/decoder/dense_33/BiasAdd/ReadVariableOp2h
2autoencoder/decoder/dense_33/MatMul/ReadVariableOp2autoencoder/decoder/dense_33/MatMul/ReadVariableOp2j
3autoencoder/decoder/dense_34/BiasAdd/ReadVariableOp3autoencoder/decoder/dense_34/BiasAdd/ReadVariableOp2h
2autoencoder/decoder/dense_34/MatMul/ReadVariableOp2autoencoder/decoder/dense_34/MatMul/ReadVariableOp2j
3autoencoder/decoder/dense_35/BiasAdd/ReadVariableOp3autoencoder/decoder/dense_35/BiasAdd/ReadVariableOp2h
2autoencoder/decoder/dense_35/MatMul/ReadVariableOp2autoencoder/decoder/dense_35/MatMul/ReadVariableOp2l
4autoencoder/encoder/conv1d_22/BiasAdd/ReadVariableOp4autoencoder/encoder/conv1d_22/BiasAdd/ReadVariableOp2
@autoencoder/encoder/conv1d_22/Conv1D/ExpandDims_1/ReadVariableOp@autoencoder/encoder/conv1d_22/Conv1D/ExpandDims_1/ReadVariableOp2l
4autoencoder/encoder/conv1d_23/BiasAdd/ReadVariableOp4autoencoder/encoder/conv1d_23/BiasAdd/ReadVariableOp2
@autoencoder/encoder/conv1d_23/Conv1D/ExpandDims_1/ReadVariableOp@autoencoder/encoder/conv1d_23/Conv1D/ExpandDims_1/ReadVariableOp2t
8autoencoder/encoder/latent_vector/BiasAdd/ReadVariableOp8autoencoder/encoder/latent_vector/BiasAdd/ReadVariableOp2r
7autoencoder/encoder/latent_vector/MatMul/ReadVariableOp7autoencoder/encoder/latent_vector/MatMul/ReadVariableOp:Z V
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
'
_user_specified_nameencoder_input
À
b
F__inference_flatten_11_layer_call_and_return_conditional_losses_861983

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ¨   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ*:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
 
_user_specified_nameinputs
É

E__inference_conv1d_23_layer_call_and_return_conditional_losses_863186

inputsA
+conv1d_expanddims_1_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :  
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:­
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
paddingVALID*
strides

Conv1D/SqueezeSqueezeConv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
squeeze_dims

ýÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*T
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*e
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ.: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.
 
_user_specified_nameinputs
Þd
Ã
G__inference_autoencoder_layer_call_and_return_conditional_losses_862891

inputsS
=encoder_conv1d_22_conv1d_expanddims_1_readvariableop_resource:?
1encoder_conv1d_22_biasadd_readvariableop_resource:S
=encoder_conv1d_23_conv1d_expanddims_1_readvariableop_resource:?
1encoder_conv1d_23_biasadd_readvariableop_resource:G
4encoder_latent_vector_matmul_readvariableop_resource:	¨@C
5encoder_latent_vector_biasadd_readvariableop_resource:@A
/decoder_dense_33_matmul_readvariableop_resource:@ >
0decoder_dense_33_biasadd_readvariableop_resource: A
/decoder_dense_34_matmul_readvariableop_resource:  >
0decoder_dense_34_biasadd_readvariableop_resource: A
/decoder_dense_35_matmul_readvariableop_resource: 2>
0decoder_dense_35_biasadd_readvariableop_resource:2
identity¢'decoder/dense_33/BiasAdd/ReadVariableOp¢&decoder/dense_33/MatMul/ReadVariableOp¢'decoder/dense_34/BiasAdd/ReadVariableOp¢&decoder/dense_34/MatMul/ReadVariableOp¢'decoder/dense_35/BiasAdd/ReadVariableOp¢&decoder/dense_35/MatMul/ReadVariableOp¢(encoder/conv1d_22/BiasAdd/ReadVariableOp¢4encoder/conv1d_22/Conv1D/ExpandDims_1/ReadVariableOp¢(encoder/conv1d_23/BiasAdd/ReadVariableOp¢4encoder/conv1d_23/Conv1D/ExpandDims_1/ReadVariableOp¢,encoder/latent_vector/BiasAdd/ReadVariableOp¢+encoder/latent_vector/MatMul/ReadVariableOpR
encoder/random_noise_4/ShapeShapeinputs*
T0*
_output_shapes
:n
)encoder/random_noise_4/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    p
+encoder/random_noise_4/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>Ú
9encoder/random_noise_4/random_normal/RandomStandardNormalRandomStandardNormal%encoder/random_noise_4/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
dtype0*
seed±ÿå)*
seed2¹Öß
(encoder/random_noise_4/random_normal/mulMulBencoder/random_noise_4/random_normal/RandomStandardNormal:output:04encoder/random_noise_4/random_normal/stddev:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2Å
$encoder/random_noise_4/random_normalAddV2,encoder/random_noise_4/random_normal/mul:z:02encoder/random_noise_4/random_normal/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
encoder/random_noise_4/addAddV2inputs(encoder/random_noise_4/random_normal:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2q
encoder/random_noise_4/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          
encoder/random_noise_4/MaxMaxencoder/random_noise_4/add:z:0%encoder/random_noise_4/Const:output:0*
T0*
_output_shapes
: ¤
encoder/random_noise_4/truedivRealDivencoder/random_noise_4/add:z:0#encoder/random_noise_4/Max:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2r
'encoder/conv1d_22/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿÁ
#encoder/conv1d_22/Conv1D/ExpandDims
ExpandDims"encoder/random_noise_4/truediv:z:00encoder/conv1d_22/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2¶
4encoder/conv1d_22/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp=encoder_conv1d_22_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0k
)encoder/conv1d_22/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : Ö
%encoder/conv1d_22/Conv1D/ExpandDims_1
ExpandDims<encoder/conv1d_22/Conv1D/ExpandDims_1/ReadVariableOp:value:02encoder/conv1d_22/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:ã
encoder/conv1d_22/Conv1DConv2D,encoder/conv1d_22/Conv1D/ExpandDims:output:0.encoder/conv1d_22/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.*
paddingVALID*
strides
¤
 encoder/conv1d_22/Conv1D/SqueezeSqueeze!encoder/conv1d_22/Conv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.*
squeeze_dims

ýÿÿÿÿÿÿÿÿ
(encoder/conv1d_22/BiasAdd/ReadVariableOpReadVariableOp1encoder_conv1d_22_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0·
encoder/conv1d_22/BiasAddBiasAdd)encoder/conv1d_22/Conv1D/Squeeze:output:00encoder/conv1d_22/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.x
encoder/conv1d_22/ReluRelu"encoder/conv1d_22/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.r
'encoder/conv1d_23/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿÃ
#encoder/conv1d_23/Conv1D/ExpandDims
ExpandDims$encoder/conv1d_22/Relu:activations:00encoder/conv1d_23/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.¶
4encoder/conv1d_23/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp=encoder_conv1d_23_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0k
)encoder/conv1d_23/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : Ö
%encoder/conv1d_23/Conv1D/ExpandDims_1
ExpandDims<encoder/conv1d_23/Conv1D/ExpandDims_1/ReadVariableOp:value:02encoder/conv1d_23/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:ã
encoder/conv1d_23/Conv1DConv2D,encoder/conv1d_23/Conv1D/ExpandDims:output:0.encoder/conv1d_23/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
paddingVALID*
strides
¤
 encoder/conv1d_23/Conv1D/SqueezeSqueeze!encoder/conv1d_23/Conv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
squeeze_dims

ýÿÿÿÿÿÿÿÿ
(encoder/conv1d_23/BiasAdd/ReadVariableOpReadVariableOp1encoder_conv1d_23_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0·
encoder/conv1d_23/BiasAddBiasAdd)encoder/conv1d_23/Conv1D/Squeeze:output:00encoder/conv1d_23/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*x
encoder/conv1d_23/ReluRelu"encoder/conv1d_23/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*i
encoder/flatten_11/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ¨   ¡
encoder/flatten_11/ReshapeReshape$encoder/conv1d_23/Relu:activations:0!encoder/flatten_11/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨¡
+encoder/latent_vector/MatMul/ReadVariableOpReadVariableOp4encoder_latent_vector_matmul_readvariableop_resource*
_output_shapes
:	¨@*
dtype0²
encoder/latent_vector/MatMulMatMul#encoder/flatten_11/Reshape:output:03encoder/latent_vector/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
,encoder/latent_vector/BiasAdd/ReadVariableOpReadVariableOp5encoder_latent_vector_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¸
encoder/latent_vector/BiasAddBiasAdd&encoder/latent_vector/MatMul:product:04encoder/latent_vector/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
&decoder/dense_33/MatMul/ReadVariableOpReadVariableOp/decoder_dense_33_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype0«
decoder/dense_33/MatMulMatMul&encoder/latent_vector/BiasAdd:output:0.decoder/dense_33/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
'decoder/dense_33/BiasAdd/ReadVariableOpReadVariableOp0decoder_dense_33_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0©
decoder/dense_33/BiasAddBiasAdd!decoder/dense_33/MatMul:product:0/decoder/dense_33/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ r
decoder/dense_33/ReluRelu!decoder/dense_33/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
&decoder/dense_34/MatMul/ReadVariableOpReadVariableOp/decoder_dense_34_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0¨
decoder/dense_34/MatMulMatMul#decoder/dense_33/Relu:activations:0.decoder/dense_34/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
'decoder/dense_34/BiasAdd/ReadVariableOpReadVariableOp0decoder_dense_34_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0©
decoder/dense_34/BiasAddBiasAdd!decoder/dense_34/MatMul:product:0/decoder/dense_34/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ r
decoder/dense_34/ReluRelu!decoder/dense_34/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
&decoder/dense_35/MatMul/ReadVariableOpReadVariableOp/decoder_dense_35_matmul_readvariableop_resource*
_output_shapes

: 2*
dtype0¨
decoder/dense_35/MatMulMatMul#decoder/dense_34/Relu:activations:0.decoder/dense_35/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
'decoder/dense_35/BiasAdd/ReadVariableOpReadVariableOp0decoder_dense_35_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0©
decoder/dense_35/BiasAddBiasAdd!decoder/dense_35/MatMul:product:0/decoder/dense_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2i
decoder/reshape_11/ShapeShape!decoder/dense_35/BiasAdd:output:0*
T0*
_output_shapes
:p
&decoder/reshape_11/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(decoder/reshape_11/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(decoder/reshape_11/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:°
 decoder/reshape_11/strided_sliceStridedSlice!decoder/reshape_11/Shape:output:0/decoder/reshape_11/strided_slice/stack:output:01decoder/reshape_11/strided_slice/stack_1:output:01decoder/reshape_11/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"decoder/reshape_11/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2d
"decoder/reshape_11/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Û
 decoder/reshape_11/Reshape/shapePack)decoder/reshape_11/strided_slice:output:0+decoder/reshape_11/Reshape/shape/1:output:0+decoder/reshape_11/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:©
decoder/reshape_11/ReshapeReshape!decoder/dense_35/BiasAdd:output:0)decoder/reshape_11/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2v
IdentityIdentity#decoder/reshape_11/Reshape:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2à
NoOpNoOp(^decoder/dense_33/BiasAdd/ReadVariableOp'^decoder/dense_33/MatMul/ReadVariableOp(^decoder/dense_34/BiasAdd/ReadVariableOp'^decoder/dense_34/MatMul/ReadVariableOp(^decoder/dense_35/BiasAdd/ReadVariableOp'^decoder/dense_35/MatMul/ReadVariableOp)^encoder/conv1d_22/BiasAdd/ReadVariableOp5^encoder/conv1d_22/Conv1D/ExpandDims_1/ReadVariableOp)^encoder/conv1d_23/BiasAdd/ReadVariableOp5^encoder/conv1d_23/Conv1D/ExpandDims_1/ReadVariableOp-^encoder/latent_vector/BiasAdd/ReadVariableOp,^encoder/latent_vector/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ2: : : : : : : : : : : : 2R
'decoder/dense_33/BiasAdd/ReadVariableOp'decoder/dense_33/BiasAdd/ReadVariableOp2P
&decoder/dense_33/MatMul/ReadVariableOp&decoder/dense_33/MatMul/ReadVariableOp2R
'decoder/dense_34/BiasAdd/ReadVariableOp'decoder/dense_34/BiasAdd/ReadVariableOp2P
&decoder/dense_34/MatMul/ReadVariableOp&decoder/dense_34/MatMul/ReadVariableOp2R
'decoder/dense_35/BiasAdd/ReadVariableOp'decoder/dense_35/BiasAdd/ReadVariableOp2P
&decoder/dense_35/MatMul/ReadVariableOp&decoder/dense_35/MatMul/ReadVariableOp2T
(encoder/conv1d_22/BiasAdd/ReadVariableOp(encoder/conv1d_22/BiasAdd/ReadVariableOp2l
4encoder/conv1d_22/Conv1D/ExpandDims_1/ReadVariableOp4encoder/conv1d_22/Conv1D/ExpandDims_1/ReadVariableOp2T
(encoder/conv1d_23/BiasAdd/ReadVariableOp(encoder/conv1d_23/BiasAdd/ReadVariableOp2l
4encoder/conv1d_23/Conv1D/ExpandDims_1/ReadVariableOp4encoder/conv1d_23/Conv1D/ExpandDims_1/ReadVariableOp2\
,encoder/latent_vector/BiasAdd/ReadVariableOp,encoder/latent_vector/BiasAdd/ReadVariableOp2Z
+encoder/latent_vector/MatMul/ReadVariableOp+encoder/latent_vector/MatMul/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
 
_user_specified_nameinputs

h
/__inference_random_noise_4_layer_call_fn_863122

inputs
identity¢StatefulPartitionedCallÉ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_random_noise_4_layer_call_and_return_conditional_losses_861931s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ222
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
 
_user_specified_nameinputs
©
G
+__inference_reshape_11_layer_call_fn_863280

inputs
identityµ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_reshape_11_layer_call_and_return_conditional_losses_862243d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ2:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
 
_user_specified_nameinputs


õ
D__inference_dense_34_layer_call_and_return_conditional_losses_863256

inputs0
matmul_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Â

)__inference_dense_33_layer_call_fn_863225

inputs
unknown:@ 
	unknown_0: 
identity¢StatefulPartitionedCallÙ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_33_layer_call_and_return_conditional_losses_862191o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ø

*__inference_conv1d_23_layer_call_fn_863170

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCallÞ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_conv1d_23_layer_call_and_return_conditional_losses_861971s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ.: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.
 
_user_specified_nameinputs
«
G
+__inference_flatten_11_layer_call_fn_863191

inputs
identity²
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_flatten_11_layer_call_and_return_conditional_losses_861983a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ*:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
 
_user_specified_nameinputs
®
i
J__inference_random_noise_4_layer_call_and_return_conditional_losses_861931

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:W
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    Y
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>¬
"random_normal/RandomStandardNormalRandomStandardNormalShape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
dtype0*
seed±ÿå)*
seed2±ëÒ
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
random_normalAddV2random_normal/mul:z:0random_normal/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2]
addAddV2inputsrandom_normal:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2Z
ConstConst*
_output_shapes
:*
dtype0*!
valueB"          C
MaxMaxinputsConst:output:0*
T0*
_output_shapes
: ^
truedivRealDivinputsMax:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2W
IdentityIdentitytruediv:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ2:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
 
_user_specified_nameinputs
¾
³
G__inference_autoencoder_layer_call_and_return_conditional_losses_862442

inputs$
encoder_862415:
encoder_862417:$
encoder_862419:
encoder_862421:!
encoder_862423:	¨@
encoder_862425:@ 
decoder_862428:@ 
decoder_862430:  
decoder_862432:  
decoder_862434:  
decoder_862436: 2
decoder_862438:2
identity¢decoder/StatefulPartitionedCall¢encoder/StatefulPartitionedCall´
encoder/StatefulPartitionedCallStatefulPartitionedCallinputsencoder_862415encoder_862417encoder_862419encoder_862421encoder_862423encoder_862425*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_encoder_layer_call_and_return_conditional_losses_862002Ú
decoder/StatefulPartitionedCallStatefulPartitionedCall(encoder/StatefulPartitionedCall:output:0decoder_862428decoder_862430decoder_862432decoder_862434decoder_862436decoder_862438*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_decoder_layer_call_and_return_conditional_losses_862246{
IdentityIdentity(decoder/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
NoOpNoOp ^decoder/StatefulPartitionedCall ^encoder/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ2: : : : : : : : : : : : 2B
decoder/StatefulPartitionedCalldecoder/StatefulPartitionedCall2B
encoder/StatefulPartitionedCallencoder/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
 
_user_specified_nameinputs
Ó
º
G__inference_autoencoder_layer_call_and_return_conditional_losses_862616
encoder_input$
encoder_862589:
encoder_862591:$
encoder_862593:
encoder_862595:!
encoder_862597:	¨@
encoder_862599:@ 
decoder_862602:@ 
decoder_862604:  
decoder_862606:  
decoder_862608:  
decoder_862610: 2
decoder_862612:2
identity¢decoder/StatefulPartitionedCall¢encoder/StatefulPartitionedCall»
encoder/StatefulPartitionedCallStatefulPartitionedCallencoder_inputencoder_862589encoder_862591encoder_862593encoder_862595encoder_862597encoder_862599*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_encoder_layer_call_and_return_conditional_losses_862002Ú
decoder/StatefulPartitionedCallStatefulPartitionedCall(encoder/StatefulPartitionedCall:output:0decoder_862602decoder_862604decoder_862606decoder_862608decoder_862610decoder_862612*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_decoder_layer_call_and_return_conditional_losses_862246{
IdentityIdentity(decoder/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
NoOpNoOp ^decoder/StatefulPartitionedCall ^encoder/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ2: : : : : : : : : : : : 2B
decoder/StatefulPartitionedCalldecoder/StatefulPartitionedCall2B
encoder/StatefulPartitionedCallencoder/StatefulPartitionedCall:Z V
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
'
_user_specified_nameencoder_input
Ð	
û
I__inference_latent_vector_layer_call_and_return_conditional_losses_861995

inputs1
matmul_readvariableop_resource:	¨@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	¨@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ¨: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
 
_user_specified_nameinputs
Ø

b
F__inference_reshape_11_layer_call_and_return_conditional_losses_863293

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2Q
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:h
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2\
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ2:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
 
_user_specified_nameinputs
ð

(__inference_decoder_layer_call_fn_863034

inputs
unknown:@ 
	unknown_0: 
	unknown_1:  
	unknown_2: 
	unknown_3: 2
	unknown_4:2
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_decoder_layer_call_and_return_conditional_losses_862246s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ@: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

±
,__inference_autoencoder_layer_call_fn_862712

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:	¨@
	unknown_4:@
	unknown_5:@ 
	unknown_6: 
	unknown_7:  
	unknown_8: 
	unknown_9: 2

unknown_10:2
identity¢StatefulPartitionedCallã
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_autoencoder_layer_call_and_return_conditional_losses_862442s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ2: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
 
_user_specified_nameinputs
ÿ
ß
C__inference_encoder_layer_call_and_return_conditional_losses_862152
encoder_input&
conv1d_22_862135:
conv1d_22_862137:&
conv1d_23_862140:
conv1d_23_862142:'
latent_vector_862146:	¨@"
latent_vector_862148:@
identity¢!conv1d_22/StatefulPartitionedCall¢!conv1d_23/StatefulPartitionedCall¢%latent_vector/StatefulPartitionedCall¢&random_noise_4/StatefulPartitionedCallß
&random_noise_4/StatefulPartitionedCallStatefulPartitionedCallencoder_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_random_noise_4_layer_call_and_return_conditional_losses_861931¡
!conv1d_22/StatefulPartitionedCallStatefulPartitionedCall/random_noise_4/StatefulPartitionedCall:output:0conv1d_22_862135conv1d_22_862137*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_conv1d_22_layer_call_and_return_conditional_losses_861949
!conv1d_23/StatefulPartitionedCallStatefulPartitionedCall*conv1d_22/StatefulPartitionedCall:output:0conv1d_23_862140conv1d_23_862142*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_conv1d_23_layer_call_and_return_conditional_losses_861971á
flatten_11/PartitionedCallPartitionedCall*conv1d_23/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_flatten_11_layer_call_and_return_conditional_losses_861983¡
%latent_vector/StatefulPartitionedCallStatefulPartitionedCall#flatten_11/PartitionedCall:output:0latent_vector_862146latent_vector_862148*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_latent_vector_layer_call_and_return_conditional_losses_861995}
IdentityIdentity.latent_vector/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ß
NoOpNoOp"^conv1d_22/StatefulPartitionedCall"^conv1d_23/StatefulPartitionedCall&^latent_vector/StatefulPartitionedCall'^random_noise_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ2: : : : : : 2F
!conv1d_22/StatefulPartitionedCall!conv1d_22/StatefulPartitionedCall2F
!conv1d_23/StatefulPartitionedCall!conv1d_23/StatefulPartitionedCall2N
%latent_vector/StatefulPartitionedCall%latent_vector/StatefulPartitionedCall2P
&random_noise_4/StatefulPartitionedCall&random_noise_4/StatefulPartitionedCall:Z V
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
'
_user_specified_nameencoder_input
	

(__inference_encoder_layer_call_fn_862017
encoder_input
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:	¨@
	unknown_4:@
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallencoder_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_encoder_layer_call_and_return_conditional_losses_862002o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ2: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
'
_user_specified_nameencoder_input
ê
Ø
C__inference_encoder_layer_call_and_return_conditional_losses_862099

inputs&
conv1d_22_862082:
conv1d_22_862084:&
conv1d_23_862087:
conv1d_23_862089:'
latent_vector_862093:	¨@"
latent_vector_862095:@
identity¢!conv1d_22/StatefulPartitionedCall¢!conv1d_23/StatefulPartitionedCall¢%latent_vector/StatefulPartitionedCall¢&random_noise_4/StatefulPartitionedCallØ
&random_noise_4/StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_random_noise_4_layer_call_and_return_conditional_losses_861931¡
!conv1d_22/StatefulPartitionedCallStatefulPartitionedCall/random_noise_4/StatefulPartitionedCall:output:0conv1d_22_862082conv1d_22_862084*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_conv1d_22_layer_call_and_return_conditional_losses_861949
!conv1d_23/StatefulPartitionedCallStatefulPartitionedCall*conv1d_22/StatefulPartitionedCall:output:0conv1d_23_862087conv1d_23_862089*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_conv1d_23_layer_call_and_return_conditional_losses_861971á
flatten_11/PartitionedCallPartitionedCall*conv1d_23/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_flatten_11_layer_call_and_return_conditional_losses_861983¡
%latent_vector/StatefulPartitionedCallStatefulPartitionedCall#flatten_11/PartitionedCall:output:0latent_vector_862093latent_vector_862095*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_latent_vector_layer_call_and_return_conditional_losses_861995}
IdentityIdentity.latent_vector/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ß
NoOpNoOp"^conv1d_22/StatefulPartitionedCall"^conv1d_23/StatefulPartitionedCall&^latent_vector/StatefulPartitionedCall'^random_noise_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ2: : : : : : 2F
!conv1d_22/StatefulPartitionedCall!conv1d_22/StatefulPartitionedCall2F
!conv1d_23/StatefulPartitionedCall!conv1d_23/StatefulPartitionedCall2N
%latent_vector/StatefulPartitionedCall%latent_vector/StatefulPartitionedCall2P
&random_noise_4/StatefulPartitionedCall&random_noise_4/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
 
_user_specified_nameinputs
î

°
$__inference_signature_wrapper_862683
encoder_input
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:	¨@
	unknown_4:@
	unknown_5:@ 
	unknown_6: 
	unknown_7:  
	unknown_8: 
	unknown_9: 2

unknown_10:2
identity¢StatefulPartitionedCallÄ
StatefulPartitionedCallStatefulPartitionedCallencoder_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__wrapped_model_861910s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ2: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
'
_user_specified_nameencoder_input
	

(__inference_decoder_layer_call_fn_862261
decoder_input
unknown:@ 
	unknown_0: 
	unknown_1:  
	unknown_2: 
	unknown_3: 2
	unknown_4:2
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCalldecoder_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_decoder_layer_call_and_return_conditional_losses_862246s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ@: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
'
_user_specified_namedecoder_input
ÿ
ß
C__inference_encoder_layer_call_and_return_conditional_losses_862173
encoder_input&
conv1d_22_862156:
conv1d_22_862158:&
conv1d_23_862161:
conv1d_23_862163:'
latent_vector_862167:	¨@"
latent_vector_862169:@
identity¢!conv1d_22/StatefulPartitionedCall¢!conv1d_23/StatefulPartitionedCall¢%latent_vector/StatefulPartitionedCall¢&random_noise_4/StatefulPartitionedCallß
&random_noise_4/StatefulPartitionedCallStatefulPartitionedCallencoder_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_random_noise_4_layer_call_and_return_conditional_losses_861931¡
!conv1d_22/StatefulPartitionedCallStatefulPartitionedCall/random_noise_4/StatefulPartitionedCall:output:0conv1d_22_862156conv1d_22_862158*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_conv1d_22_layer_call_and_return_conditional_losses_861949
!conv1d_23/StatefulPartitionedCallStatefulPartitionedCall*conv1d_22/StatefulPartitionedCall:output:0conv1d_23_862161conv1d_23_862163*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_conv1d_23_layer_call_and_return_conditional_losses_861971á
flatten_11/PartitionedCallPartitionedCall*conv1d_23/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_flatten_11_layer_call_and_return_conditional_losses_861983¡
%latent_vector/StatefulPartitionedCallStatefulPartitionedCall#flatten_11/PartitionedCall:output:0latent_vector_862167latent_vector_862169*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_latent_vector_layer_call_and_return_conditional_losses_861995}
IdentityIdentity.latent_vector/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ß
NoOpNoOp"^conv1d_22/StatefulPartitionedCall"^conv1d_23/StatefulPartitionedCall&^latent_vector/StatefulPartitionedCall'^random_noise_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ2: : : : : : 2F
!conv1d_22/StatefulPartitionedCall!conv1d_22/StatefulPartitionedCall2F
!conv1d_23/StatefulPartitionedCall!conv1d_23/StatefulPartitionedCall2N
%latent_vector/StatefulPartitionedCall%latent_vector/StatefulPartitionedCall2P
&random_noise_4/StatefulPartitionedCall&random_noise_4/StatefulPartitionedCall:Z V
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
'
_user_specified_nameencoder_input
¾
³
G__inference_autoencoder_layer_call_and_return_conditional_losses_862530

inputs$
encoder_862503:
encoder_862505:$
encoder_862507:
encoder_862509:!
encoder_862511:	¨@
encoder_862513:@ 
decoder_862516:@ 
decoder_862518:  
decoder_862520:  
decoder_862522:  
decoder_862524: 2
decoder_862526:2
identity¢decoder/StatefulPartitionedCall¢encoder/StatefulPartitionedCall´
encoder/StatefulPartitionedCallStatefulPartitionedCallinputsencoder_862503encoder_862505encoder_862507encoder_862509encoder_862511encoder_862513*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_encoder_layer_call_and_return_conditional_losses_862099Ú
decoder/StatefulPartitionedCallStatefulPartitionedCall(encoder/StatefulPartitionedCall:output:0decoder_862516decoder_862518decoder_862520decoder_862522decoder_862524decoder_862526*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_decoder_layer_call_and_return_conditional_losses_862336{
IdentityIdentity(decoder/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
NoOpNoOp ^decoder/StatefulPartitionedCall ^encoder/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ2: : : : : : : : : : : : 2B
decoder/StatefulPartitionedCalldecoder/StatefulPartitionedCall2B
encoder/StatefulPartitionedCallencoder/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
 
_user_specified_nameinputs
ø

C__inference_decoder_layer_call_and_return_conditional_losses_862388
decoder_input!
dense_33_862371:@ 
dense_33_862373: !
dense_34_862376:  
dense_34_862378: !
dense_35_862381: 2
dense_35_862383:2
identity¢ dense_33/StatefulPartitionedCall¢ dense_34/StatefulPartitionedCall¢ dense_35/StatefulPartitionedCall÷
 dense_33/StatefulPartitionedCallStatefulPartitionedCalldecoder_inputdense_33_862371dense_33_862373*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_33_layer_call_and_return_conditional_losses_862191
 dense_34/StatefulPartitionedCallStatefulPartitionedCall)dense_33/StatefulPartitionedCall:output:0dense_34_862376dense_34_862378*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_34_layer_call_and_return_conditional_losses_862208
 dense_35/StatefulPartitionedCallStatefulPartitionedCall)dense_34/StatefulPartitionedCall:output:0dense_35_862381dense_35_862383*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_35_layer_call_and_return_conditional_losses_862224ã
reshape_11/PartitionedCallPartitionedCall)dense_35/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_reshape_11_layer_call_and_return_conditional_losses_862243v
IdentityIdentity#reshape_11/PartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2¯
NoOpNoOp!^dense_33/StatefulPartitionedCall!^dense_34/StatefulPartitionedCall!^dense_35/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ@: : : : : : 2D
 dense_33/StatefulPartitionedCall dense_33/StatefulPartitionedCall2D
 dense_34/StatefulPartitionedCall dense_34/StatefulPartitionedCall2D
 dense_35/StatefulPartitionedCall dense_35/StatefulPartitionedCall:V R
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
'
_user_specified_namedecoder_input
ø

C__inference_decoder_layer_call_and_return_conditional_losses_862408
decoder_input!
dense_33_862391:@ 
dense_33_862393: !
dense_34_862396:  
dense_34_862398: !
dense_35_862401: 2
dense_35_862403:2
identity¢ dense_33/StatefulPartitionedCall¢ dense_34/StatefulPartitionedCall¢ dense_35/StatefulPartitionedCall÷
 dense_33/StatefulPartitionedCallStatefulPartitionedCalldecoder_inputdense_33_862391dense_33_862393*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_33_layer_call_and_return_conditional_losses_862191
 dense_34/StatefulPartitionedCallStatefulPartitionedCall)dense_33/StatefulPartitionedCall:output:0dense_34_862396dense_34_862398*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_34_layer_call_and_return_conditional_losses_862208
 dense_35/StatefulPartitionedCallStatefulPartitionedCall)dense_34/StatefulPartitionedCall:output:0dense_35_862401dense_35_862403*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_35_layer_call_and_return_conditional_losses_862224ã
reshape_11/PartitionedCallPartitionedCall)dense_35/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_reshape_11_layer_call_and_return_conditional_losses_862243v
IdentityIdentity#reshape_11/PartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2¯
NoOpNoOp!^dense_33/StatefulPartitionedCall!^dense_34/StatefulPartitionedCall!^dense_35/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ@: : : : : : 2D
 dense_33/StatefulPartitionedCall dense_33/StatefulPartitionedCall2D
 dense_34/StatefulPartitionedCall dense_34/StatefulPartitionedCall2D
 dense_35/StatefulPartitionedCall dense_35/StatefulPartitionedCall:V R
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
'
_user_specified_namedecoder_input
Ï

.__inference_latent_vector_layer_call_fn_863206

inputs
unknown:	¨@
	unknown_0:@
identity¢StatefulPartitionedCallÞ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_latent_vector_layer_call_and_return_conditional_losses_861995o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ¨: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
 
_user_specified_nameinputs
ð

(__inference_decoder_layer_call_fn_863051

inputs
unknown:@ 
	unknown_0: 
	unknown_1:  
	unknown_2: 
	unknown_3: 2
	unknown_4:2
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_decoder_layer_call_and_return_conditional_losses_862336s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ@: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ç	
õ
D__inference_dense_35_layer_call_and_return_conditional_losses_862224

inputs0
matmul_readvariableop_resource: 2-
biasadd_readvariableop_resource:2
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: 2*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ç	
õ
D__inference_dense_35_layer_call_and_return_conditional_losses_863275

inputs0
matmul_readvariableop_resource: 2-
biasadd_readvariableop_resource:2
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: 2*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
®
i
J__inference_random_noise_4_layer_call_and_return_conditional_losses_863136

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:W
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    Y
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>¬
"random_normal/RandomStandardNormalRandomStandardNormalShape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
dtype0*
seed±ÿå)*
seed2§
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
random_normalAddV2random_normal/mul:z:0random_normal/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2]
addAddV2inputsrandom_normal:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2Z
ConstConst*
_output_shapes
:*
dtype0*!
valueB"          C
MaxMaxinputsConst:output:0*
T0*
_output_shapes
: ^
truedivRealDivinputsMax:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2W
IdentityIdentitytruediv:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ2:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
 
_user_specified_nameinputs
É

E__inference_conv1d_22_layer_call_and_return_conditional_losses_863161

inputsA
+conv1d_expanddims_1_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :  
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:­
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.*
paddingVALID*
strides

Conv1D/SqueezeSqueezeConv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.*
squeeze_dims

ýÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.T
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.e
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ2: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
 
_user_specified_nameinputs
ù

(__inference_encoder_layer_call_fn_862908

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:	¨@
	unknown_4:@
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_encoder_layer_call_and_return_conditional_losses_862002o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ2: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
 
_user_specified_nameinputs
É

E__inference_conv1d_23_layer_call_and_return_conditional_losses_861971

inputsA
+conv1d_expanddims_1_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :  
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:­
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
paddingVALID*
strides

Conv1D/SqueezeSqueezeConv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
squeeze_dims

ýÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*T
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*e
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ.: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.
 
_user_specified_nameinputs
ö$

C__inference_decoder_layer_call_and_return_conditional_losses_863117

inputs9
'dense_33_matmul_readvariableop_resource:@ 6
(dense_33_biasadd_readvariableop_resource: 9
'dense_34_matmul_readvariableop_resource:  6
(dense_34_biasadd_readvariableop_resource: 9
'dense_35_matmul_readvariableop_resource: 26
(dense_35_biasadd_readvariableop_resource:2
identity¢dense_33/BiasAdd/ReadVariableOp¢dense_33/MatMul/ReadVariableOp¢dense_34/BiasAdd/ReadVariableOp¢dense_34/MatMul/ReadVariableOp¢dense_35/BiasAdd/ReadVariableOp¢dense_35/MatMul/ReadVariableOp
dense_33/MatMul/ReadVariableOpReadVariableOp'dense_33_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype0{
dense_33/MatMulMatMulinputs&dense_33/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
dense_33/BiasAdd/ReadVariableOpReadVariableOp(dense_33_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
dense_33/BiasAddBiasAdddense_33/MatMul:product:0'dense_33/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ b
dense_33/ReluReludense_33/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
dense_34/MatMul/ReadVariableOpReadVariableOp'dense_34_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0
dense_34/MatMulMatMuldense_33/Relu:activations:0&dense_34/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
dense_34/BiasAdd/ReadVariableOpReadVariableOp(dense_34_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
dense_34/BiasAddBiasAdddense_34/MatMul:product:0'dense_34/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ b
dense_34/ReluReludense_34/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
dense_35/MatMul/ReadVariableOpReadVariableOp'dense_35_matmul_readvariableop_resource*
_output_shapes

: 2*
dtype0
dense_35/MatMulMatMuldense_34/Relu:activations:0&dense_35/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_35/BiasAdd/ReadVariableOpReadVariableOp(dense_35_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0
dense_35/BiasAddBiasAdddense_35/MatMul:product:0'dense_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2Y
reshape_11/ShapeShapedense_35/BiasAdd:output:0*
T0*
_output_shapes
:h
reshape_11/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: j
 reshape_11/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j
 reshape_11/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
reshape_11/strided_sliceStridedSlicereshape_11/Shape:output:0'reshape_11/strided_slice/stack:output:0)reshape_11/strided_slice/stack_1:output:0)reshape_11/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
reshape_11/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2\
reshape_11/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :»
reshape_11/Reshape/shapePack!reshape_11/strided_slice:output:0#reshape_11/Reshape/shape/1:output:0#reshape_11/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:
reshape_11/ReshapeReshapedense_35/BiasAdd:output:0!reshape_11/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2n
IdentityIdentityreshape_11/Reshape:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
NoOpNoOp ^dense_33/BiasAdd/ReadVariableOp^dense_33/MatMul/ReadVariableOp ^dense_34/BiasAdd/ReadVariableOp^dense_34/MatMul/ReadVariableOp ^dense_35/BiasAdd/ReadVariableOp^dense_35/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ@: : : : : : 2B
dense_33/BiasAdd/ReadVariableOpdense_33/BiasAdd/ReadVariableOp2@
dense_33/MatMul/ReadVariableOpdense_33/MatMul/ReadVariableOp2B
dense_34/BiasAdd/ReadVariableOpdense_34/BiasAdd/ReadVariableOp2@
dense_34/MatMul/ReadVariableOpdense_34/MatMul/ReadVariableOp2B
dense_35/BiasAdd/ReadVariableOpdense_35/BiasAdd/ReadVariableOp2@
dense_35/MatMul/ReadVariableOpdense_35/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
üT

__inference__traced_save_863445
file_prefix/
+savev2_conv1d_22_kernel_read_readvariableop-
)savev2_conv1d_22_bias_read_readvariableop/
+savev2_conv1d_23_kernel_read_readvariableop-
)savev2_conv1d_23_bias_read_readvariableop3
/savev2_latent_vector_kernel_read_readvariableop1
-savev2_latent_vector_bias_read_readvariableop.
*savev2_dense_33_kernel_read_readvariableop,
(savev2_dense_33_bias_read_readvariableop.
*savev2_dense_34_kernel_read_readvariableop,
(savev2_dense_34_bias_read_readvariableop.
*savev2_dense_35_kernel_read_readvariableop,
(savev2_dense_35_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop6
2savev2_adam_conv1d_22_kernel_m_read_readvariableop4
0savev2_adam_conv1d_22_bias_m_read_readvariableop6
2savev2_adam_conv1d_23_kernel_m_read_readvariableop4
0savev2_adam_conv1d_23_bias_m_read_readvariableop:
6savev2_adam_latent_vector_kernel_m_read_readvariableop8
4savev2_adam_latent_vector_bias_m_read_readvariableop5
1savev2_adam_dense_33_kernel_m_read_readvariableop3
/savev2_adam_dense_33_bias_m_read_readvariableop5
1savev2_adam_dense_34_kernel_m_read_readvariableop3
/savev2_adam_dense_34_bias_m_read_readvariableop5
1savev2_adam_dense_35_kernel_m_read_readvariableop3
/savev2_adam_dense_35_bias_m_read_readvariableop6
2savev2_adam_conv1d_22_kernel_v_read_readvariableop4
0savev2_adam_conv1d_22_bias_v_read_readvariableop6
2savev2_adam_conv1d_23_kernel_v_read_readvariableop4
0savev2_adam_conv1d_23_bias_v_read_readvariableop:
6savev2_adam_latent_vector_kernel_v_read_readvariableop8
4savev2_adam_latent_vector_bias_v_read_readvariableop5
1savev2_adam_dense_33_kernel_v_read_readvariableop3
/savev2_adam_dense_33_bias_v_read_readvariableop5
1savev2_adam_dense_34_kernel_v_read_readvariableop3
/savev2_adam_dense_34_bias_v_read_readvariableop5
1savev2_adam_dense_35_kernel_v_read_readvariableop3
/savev2_adam_dense_35_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ¡
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:,*
dtype0*Ê
valueÀB½,B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHÅ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:,*
dtype0*k
valuebB`,B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B É
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv1d_22_kernel_read_readvariableop)savev2_conv1d_22_bias_read_readvariableop+savev2_conv1d_23_kernel_read_readvariableop)savev2_conv1d_23_bias_read_readvariableop/savev2_latent_vector_kernel_read_readvariableop-savev2_latent_vector_bias_read_readvariableop*savev2_dense_33_kernel_read_readvariableop(savev2_dense_33_bias_read_readvariableop*savev2_dense_34_kernel_read_readvariableop(savev2_dense_34_bias_read_readvariableop*savev2_dense_35_kernel_read_readvariableop(savev2_dense_35_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop2savev2_adam_conv1d_22_kernel_m_read_readvariableop0savev2_adam_conv1d_22_bias_m_read_readvariableop2savev2_adam_conv1d_23_kernel_m_read_readvariableop0savev2_adam_conv1d_23_bias_m_read_readvariableop6savev2_adam_latent_vector_kernel_m_read_readvariableop4savev2_adam_latent_vector_bias_m_read_readvariableop1savev2_adam_dense_33_kernel_m_read_readvariableop/savev2_adam_dense_33_bias_m_read_readvariableop1savev2_adam_dense_34_kernel_m_read_readvariableop/savev2_adam_dense_34_bias_m_read_readvariableop1savev2_adam_dense_35_kernel_m_read_readvariableop/savev2_adam_dense_35_bias_m_read_readvariableop2savev2_adam_conv1d_22_kernel_v_read_readvariableop0savev2_adam_conv1d_22_bias_v_read_readvariableop2savev2_adam_conv1d_23_kernel_v_read_readvariableop0savev2_adam_conv1d_23_bias_v_read_readvariableop6savev2_adam_latent_vector_kernel_v_read_readvariableop4savev2_adam_latent_vector_bias_v_read_readvariableop1savev2_adam_dense_33_kernel_v_read_readvariableop/savev2_adam_dense_33_bias_v_read_readvariableop1savev2_adam_dense_34_kernel_v_read_readvariableop/savev2_adam_dense_34_bias_v_read_readvariableop1savev2_adam_dense_35_kernel_v_read_readvariableop/savev2_adam_dense_35_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *:
dtypes0
.2,	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*â
_input_shapesÐ
Í: :::::	¨@:@:@ : :  : : 2:2: : : : : : : :::::	¨@:@:@ : :  : : 2:2:::::	¨@:@:@ : :  : : 2:2: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:($
"
_output_shapes
:: 

_output_shapes
::($
"
_output_shapes
:: 

_output_shapes
::%!

_output_shapes
:	¨@: 

_output_shapes
:@:$ 

_output_shapes

:@ : 

_output_shapes
: :$	 

_output_shapes

:  : 


_output_shapes
: :$ 

_output_shapes

: 2: 

_output_shapes
:2:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :($
"
_output_shapes
:: 

_output_shapes
::($
"
_output_shapes
:: 

_output_shapes
::%!

_output_shapes
:	¨@: 

_output_shapes
:@:$ 

_output_shapes

:@ : 

_output_shapes
: :$ 

_output_shapes

:  : 

_output_shapes
: :$ 

_output_shapes

: 2: 

_output_shapes
:2:( $
"
_output_shapes
:: !

_output_shapes
::("$
"
_output_shapes
:: #

_output_shapes
::%$!

_output_shapes
:	¨@: %

_output_shapes
:@:$& 

_output_shapes

:@ : '

_output_shapes
: :$( 

_output_shapes

:  : )

_output_shapes
: :$* 

_output_shapes

: 2: +

_output_shapes
:2:,

_output_shapes
: 
Ø

*__inference_conv1d_22_layer_call_fn_863145

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCallÞ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_conv1d_22_layer_call_and_return_conditional_losses_861949s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ2: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
 
_user_specified_nameinputs
ê
Ø
C__inference_encoder_layer_call_and_return_conditional_losses_862002

inputs&
conv1d_22_861950:
conv1d_22_861952:&
conv1d_23_861972:
conv1d_23_861974:'
latent_vector_861996:	¨@"
latent_vector_861998:@
identity¢!conv1d_22/StatefulPartitionedCall¢!conv1d_23/StatefulPartitionedCall¢%latent_vector/StatefulPartitionedCall¢&random_noise_4/StatefulPartitionedCallØ
&random_noise_4/StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_random_noise_4_layer_call_and_return_conditional_losses_861931¡
!conv1d_22/StatefulPartitionedCallStatefulPartitionedCall/random_noise_4/StatefulPartitionedCall:output:0conv1d_22_861950conv1d_22_861952*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_conv1d_22_layer_call_and_return_conditional_losses_861949
!conv1d_23/StatefulPartitionedCallStatefulPartitionedCall*conv1d_22/StatefulPartitionedCall:output:0conv1d_23_861972conv1d_23_861974*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_conv1d_23_layer_call_and_return_conditional_losses_861971á
flatten_11/PartitionedCallPartitionedCall*conv1d_23/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_flatten_11_layer_call_and_return_conditional_losses_861983¡
%latent_vector/StatefulPartitionedCallStatefulPartitionedCall#flatten_11/PartitionedCall:output:0latent_vector_861996latent_vector_861998*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_latent_vector_layer_call_and_return_conditional_losses_861995}
IdentityIdentity.latent_vector/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ß
NoOpNoOp"^conv1d_22/StatefulPartitionedCall"^conv1d_23/StatefulPartitionedCall&^latent_vector/StatefulPartitionedCall'^random_noise_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ2: : : : : : 2F
!conv1d_22/StatefulPartitionedCall!conv1d_22/StatefulPartitionedCall2F
!conv1d_23/StatefulPartitionedCall!conv1d_23/StatefulPartitionedCall2N
%latent_vector/StatefulPartitionedCall%latent_vector/StatefulPartitionedCall2P
&random_noise_4/StatefulPartitionedCall&random_noise_4/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
 
_user_specified_nameinputs
Â

)__inference_dense_34_layer_call_fn_863245

inputs
unknown:  
	unknown_0: 
identity¢StatefulPartitionedCallÙ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_34_layer_call_and_return_conditional_losses_862208o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
É

E__inference_conv1d_22_layer_call_and_return_conditional_losses_861949

inputsA
+conv1d_expanddims_1_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :  
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:­
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.*
paddingVALID*
strides

Conv1D/SqueezeSqueezeConv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.*
squeeze_dims

ýÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.T
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.e
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ2: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
 
_user_specified_nameinputs


õ
D__inference_dense_33_layer_call_and_return_conditional_losses_863236

inputs0
matmul_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs"¿L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*¾
serving_defaultª
K
encoder_input:
serving_default_encoder_input:0ÿÿÿÿÿÿÿÿÿ2?
decoder4
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿ2tensorflow/serving/predict:ô°
¾
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*	&call_and_return_all_conditional_losses

_default_save_signature
	optimizer

signatures"
_tf_keras_network
"
_tf_keras_input_layer
Ã
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_network
¶
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer-4
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses"
_tf_keras_network
v
#0
$1
%2
&3
'4
(5
)6
*7
+8
,9
-10
.11"
trackable_list_wrapper
v
#0
$1
%2
&3
'4
(5
)6
*7
+8
,9
-10
.11"
trackable_list_wrapper
 "
trackable_list_wrapper
Ê
/non_trainable_variables

0layers
1metrics
2layer_regularization_losses
3layer_metrics
	variables
trainable_variables
regularization_losses
__call__

_default_save_signature
*	&call_and_return_all_conditional_losses
&	"call_and_return_conditional_losses"
_generic_user_object
æ
4trace_0
5trace_1
6trace_2
7trace_32û
,__inference_autoencoder_layer_call_fn_862469
,__inference_autoencoder_layer_call_fn_862712
,__inference_autoencoder_layer_call_fn_862741
,__inference_autoencoder_layer_call_fn_862586À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 z4trace_0z5trace_1z6trace_2z7trace_3
Ò
8trace_0
9trace_1
:trace_2
;trace_32ç
G__inference_autoencoder_layer_call_and_return_conditional_losses_862816
G__inference_autoencoder_layer_call_and_return_conditional_losses_862891
G__inference_autoencoder_layer_call_and_return_conditional_losses_862616
G__inference_autoencoder_layer_call_and_return_conditional_losses_862646À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 z8trace_0z9trace_1z:trace_2z;trace_3
ÒBÏ
!__inference__wrapped_model_861910encoder_input"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ã
<iter

=beta_1

>beta_2
	?decay
@learning_rate#mØ$mÙ%mÚ&mÛ'mÜ(mÝ)mÞ*mß+mà,má-mâ.mã#vä$vå%væ&vç'vè(vé)vê*vë+vì,ví-vî.vï"
	optimizer
,
Aserving_default"
signature_map
¥
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
F__call__
*G&call_and_return_all_conditional_losses"
_tf_keras_layer
Ý
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
L__call__
*M&call_and_return_all_conditional_losses

#kernel
$bias
 N_jit_compiled_convolution_op"
_tf_keras_layer
Ý
O	variables
Ptrainable_variables
Qregularization_losses
R	keras_api
S__call__
*T&call_and_return_all_conditional_losses

%kernel
&bias
 U_jit_compiled_convolution_op"
_tf_keras_layer
¥
V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
Z__call__
*[&call_and_return_all_conditional_losses"
_tf_keras_layer
»
\	variables
]trainable_variables
^regularization_losses
_	keras_api
`__call__
*a&call_and_return_all_conditional_losses

'kernel
(bias"
_tf_keras_layer
J
#0
$1
%2
&3
'4
(5"
trackable_list_wrapper
J
#0
$1
%2
&3
'4
(5"
trackable_list_wrapper
 "
trackable_list_wrapper
­
bnon_trainable_variables

clayers
dmetrics
elayer_regularization_losses
flayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Ö
gtrace_0
htrace_1
itrace_2
jtrace_32ë
(__inference_encoder_layer_call_fn_862017
(__inference_encoder_layer_call_fn_862908
(__inference_encoder_layer_call_fn_862925
(__inference_encoder_layer_call_fn_862131À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 zgtrace_0zhtrace_1zitrace_2zjtrace_3
Â
ktrace_0
ltrace_1
mtrace_2
ntrace_32×
C__inference_encoder_layer_call_and_return_conditional_losses_862971
C__inference_encoder_layer_call_and_return_conditional_losses_863017
C__inference_encoder_layer_call_and_return_conditional_losses_862152
C__inference_encoder_layer_call_and_return_conditional_losses_862173À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 zktrace_0zltrace_1zmtrace_2zntrace_3
"
_tf_keras_input_layer
»
o	variables
ptrainable_variables
qregularization_losses
r	keras_api
s__call__
*t&call_and_return_all_conditional_losses

)kernel
*bias"
_tf_keras_layer
»
u	variables
vtrainable_variables
wregularization_losses
x	keras_api
y__call__
*z&call_and_return_all_conditional_losses

+kernel
,bias"
_tf_keras_layer
¼
{	variables
|trainable_variables
}regularization_losses
~	keras_api
__call__
+&call_and_return_all_conditional_losses

-kernel
.bias"
_tf_keras_layer
«
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
J
)0
*1
+2
,3
-4
.5"
trackable_list_wrapper
J
)0
*1
+2
,3
-4
.5"
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses"
_generic_user_object
Þ
trace_0
trace_1
trace_2
trace_32ë
(__inference_decoder_layer_call_fn_862261
(__inference_decoder_layer_call_fn_863034
(__inference_decoder_layer_call_fn_863051
(__inference_decoder_layer_call_fn_862368À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 ztrace_0ztrace_1ztrace_2ztrace_3
Ê
trace_0
trace_1
trace_2
trace_32×
C__inference_decoder_layer_call_and_return_conditional_losses_863084
C__inference_decoder_layer_call_and_return_conditional_losses_863117
C__inference_decoder_layer_call_and_return_conditional_losses_862388
C__inference_decoder_layer_call_and_return_conditional_losses_862408À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 ztrace_0ztrace_1ztrace_2ztrace_3
&:$2conv1d_22/kernel
:2conv1d_22/bias
&:$2conv1d_23/kernel
:2conv1d_23/bias
':%	¨@2latent_vector/kernel
 :@2latent_vector/bias
!:@ 2dense_33/kernel
: 2dense_33/bias
!:  2dense_34/kernel
: 2dense_34/bias
!: 22dense_35/kernel
:22dense_35/bias
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
(
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
,__inference_autoencoder_layer_call_fn_862469encoder_input"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
þBû
,__inference_autoencoder_layer_call_fn_862712inputs"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
þBû
,__inference_autoencoder_layer_call_fn_862741inputs"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
,__inference_autoencoder_layer_call_fn_862586encoder_input"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
G__inference_autoencoder_layer_call_and_return_conditional_losses_862816inputs"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
G__inference_autoencoder_layer_call_and_return_conditional_losses_862891inputs"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 B
G__inference_autoencoder_layer_call_and_return_conditional_losses_862616encoder_input"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 B
G__inference_autoencoder_layer_call_and_return_conditional_losses_862646encoder_input"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
ÑBÎ
$__inference_signature_wrapper_862683encoder_input"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses"
_generic_user_object
õ
trace_02Ö
/__inference_random_noise_4_layer_call_fn_863122¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0

trace_02ñ
J__inference_random_noise_4_layer_call_and_return_conditional_losses_863136¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0
.
#0
$1"
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
 layer_metrics
H	variables
Itrainable_variables
Jregularization_losses
L__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses"
_generic_user_object
ð
¡trace_02Ñ
*__inference_conv1d_22_layer_call_fn_863145¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z¡trace_0

¢trace_02ì
E__inference_conv1d_22_layer_call_and_return_conditional_losses_863161¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z¢trace_0
´2±®
£²
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 0
.
%0
&1"
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
£non_trainable_variables
¤layers
¥metrics
 ¦layer_regularization_losses
§layer_metrics
O	variables
Ptrainable_variables
Qregularization_losses
S__call__
*T&call_and_return_all_conditional_losses
&T"call_and_return_conditional_losses"
_generic_user_object
ð
¨trace_02Ñ
*__inference_conv1d_23_layer_call_fn_863170¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z¨trace_0

©trace_02ì
E__inference_conv1d_23_layer_call_and_return_conditional_losses_863186¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z©trace_0
´2±®
£²
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
ªnon_trainable_variables
«layers
¬metrics
 ­layer_regularization_losses
®layer_metrics
V	variables
Wtrainable_variables
Xregularization_losses
Z__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses"
_generic_user_object
ñ
¯trace_02Ò
+__inference_flatten_11_layer_call_fn_863191¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z¯trace_0

°trace_02í
F__inference_flatten_11_layer_call_and_return_conditional_losses_863197¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z°trace_0
.
'0
(1"
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
±non_trainable_variables
²layers
³metrics
 ´layer_regularization_losses
µlayer_metrics
\	variables
]trainable_variables
^regularization_losses
`__call__
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses"
_generic_user_object
ô
¶trace_02Õ
.__inference_latent_vector_layer_call_fn_863206¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z¶trace_0

·trace_02ð
I__inference_latent_vector_layer_call_and_return_conditional_losses_863216¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z·trace_0
 "
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
Bþ
(__inference_encoder_layer_call_fn_862017encoder_input"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
úB÷
(__inference_encoder_layer_call_fn_862908inputs"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
úB÷
(__inference_encoder_layer_call_fn_862925inputs"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Bþ
(__inference_encoder_layer_call_fn_862131encoder_input"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
C__inference_encoder_layer_call_and_return_conditional_losses_862971inputs"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
C__inference_encoder_layer_call_and_return_conditional_losses_863017inputs"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
C__inference_encoder_layer_call_and_return_conditional_losses_862152encoder_input"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
C__inference_encoder_layer_call_and_return_conditional_losses_862173encoder_input"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
.
)0
*1"
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
¸non_trainable_variables
¹layers
ºmetrics
 »layer_regularization_losses
¼layer_metrics
o	variables
ptrainable_variables
qregularization_losses
s__call__
*t&call_and_return_all_conditional_losses
&t"call_and_return_conditional_losses"
_generic_user_object
ï
½trace_02Ð
)__inference_dense_33_layer_call_fn_863225¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z½trace_0

¾trace_02ë
D__inference_dense_33_layer_call_and_return_conditional_losses_863236¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z¾trace_0
.
+0
,1"
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
¿non_trainable_variables
Àlayers
Ámetrics
 Âlayer_regularization_losses
Ãlayer_metrics
u	variables
vtrainable_variables
wregularization_losses
y__call__
*z&call_and_return_all_conditional_losses
&z"call_and_return_conditional_losses"
_generic_user_object
ï
Ätrace_02Ð
)__inference_dense_34_layer_call_fn_863245¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zÄtrace_0

Åtrace_02ë
D__inference_dense_34_layer_call_and_return_conditional_losses_863256¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zÅtrace_0
.
-0
.1"
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
 "
trackable_list_wrapper
´
Ænon_trainable_variables
Çlayers
Èmetrics
 Élayer_regularization_losses
Êlayer_metrics
{	variables
|trainable_variables
}regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ï
Ëtrace_02Ð
)__inference_dense_35_layer_call_fn_863265¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zËtrace_0

Ìtrace_02ë
D__inference_dense_35_layer_call_and_return_conditional_losses_863275¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zÌtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ínon_trainable_variables
Îlayers
Ïmetrics
 Ðlayer_regularization_losses
Ñlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ñ
Òtrace_02Ò
+__inference_reshape_11_layer_call_fn_863280¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zÒtrace_0

Ótrace_02í
F__inference_reshape_11_layer_call_and_return_conditional_losses_863293¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zÓtrace_0
 "
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
Bþ
(__inference_decoder_layer_call_fn_862261decoder_input"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
úB÷
(__inference_decoder_layer_call_fn_863034inputs"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
úB÷
(__inference_decoder_layer_call_fn_863051inputs"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Bþ
(__inference_decoder_layer_call_fn_862368decoder_input"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
C__inference_decoder_layer_call_and_return_conditional_losses_863084inputs"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
C__inference_decoder_layer_call_and_return_conditional_losses_863117inputs"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
C__inference_decoder_layer_call_and_return_conditional_losses_862388decoder_input"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
C__inference_decoder_layer_call_and_return_conditional_losses_862408decoder_input"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
R
Ô	variables
Õ	keras_api

Ötotal

×count"
_tf_keras_metric
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ãBà
/__inference_random_noise_4_layer_call_fn_863122inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
þBû
J__inference_random_noise_4_layer_call_and_return_conditional_losses_863136inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÞBÛ
*__inference_conv1d_22_layer_call_fn_863145inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ùBö
E__inference_conv1d_22_layer_call_and_return_conditional_losses_863161inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÞBÛ
*__inference_conv1d_23_layer_call_fn_863170inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ùBö
E__inference_conv1d_23_layer_call_and_return_conditional_losses_863186inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ßBÜ
+__inference_flatten_11_layer_call_fn_863191inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
úB÷
F__inference_flatten_11_layer_call_and_return_conditional_losses_863197inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
âBß
.__inference_latent_vector_layer_call_fn_863206inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ýBú
I__inference_latent_vector_layer_call_and_return_conditional_losses_863216inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÝBÚ
)__inference_dense_33_layer_call_fn_863225inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
øBõ
D__inference_dense_33_layer_call_and_return_conditional_losses_863236inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÝBÚ
)__inference_dense_34_layer_call_fn_863245inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
øBõ
D__inference_dense_34_layer_call_and_return_conditional_losses_863256inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÝBÚ
)__inference_dense_35_layer_call_fn_863265inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
øBõ
D__inference_dense_35_layer_call_and_return_conditional_losses_863275inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ßBÜ
+__inference_reshape_11_layer_call_fn_863280inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
úB÷
F__inference_reshape_11_layer_call_and_return_conditional_losses_863293inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
0
Ö0
×1"
trackable_list_wrapper
.
Ô	variables"
_generic_user_object
:  (2total
:  (2count
+:)2Adam/conv1d_22/kernel/m
!:2Adam/conv1d_22/bias/m
+:)2Adam/conv1d_23/kernel/m
!:2Adam/conv1d_23/bias/m
,:*	¨@2Adam/latent_vector/kernel/m
%:#@2Adam/latent_vector/bias/m
&:$@ 2Adam/dense_33/kernel/m
 : 2Adam/dense_33/bias/m
&:$  2Adam/dense_34/kernel/m
 : 2Adam/dense_34/bias/m
&:$ 22Adam/dense_35/kernel/m
 :22Adam/dense_35/bias/m
+:)2Adam/conv1d_22/kernel/v
!:2Adam/conv1d_22/bias/v
+:)2Adam/conv1d_23/kernel/v
!:2Adam/conv1d_23/bias/v
,:*	¨@2Adam/latent_vector/kernel/v
%:#@2Adam/latent_vector/bias/v
&:$@ 2Adam/dense_33/kernel/v
 : 2Adam/dense_33/bias/v
&:$  2Adam/dense_34/kernel/v
 : 2Adam/dense_34/bias/v
&:$ 22Adam/dense_35/kernel/v
 :22Adam/dense_35/bias/v§
!__inference__wrapped_model_861910#$%&'()*+,-.:¢7
0¢-
+(
encoder_inputÿÿÿÿÿÿÿÿÿ2
ª "5ª2
0
decoder%"
decoderÿÿÿÿÿÿÿÿÿ2È
G__inference_autoencoder_layer_call_and_return_conditional_losses_862616}#$%&'()*+,-.B¢?
8¢5
+(
encoder_inputÿÿÿÿÿÿÿÿÿ2
p 

 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ2
 È
G__inference_autoencoder_layer_call_and_return_conditional_losses_862646}#$%&'()*+,-.B¢?
8¢5
+(
encoder_inputÿÿÿÿÿÿÿÿÿ2
p

 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ2
 Á
G__inference_autoencoder_layer_call_and_return_conditional_losses_862816v#$%&'()*+,-.;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ2
p 

 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ2
 Á
G__inference_autoencoder_layer_call_and_return_conditional_losses_862891v#$%&'()*+,-.;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ2
p

 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ2
  
,__inference_autoencoder_layer_call_fn_862469p#$%&'()*+,-.B¢?
8¢5
+(
encoder_inputÿÿÿÿÿÿÿÿÿ2
p 

 
ª "ÿÿÿÿÿÿÿÿÿ2 
,__inference_autoencoder_layer_call_fn_862586p#$%&'()*+,-.B¢?
8¢5
+(
encoder_inputÿÿÿÿÿÿÿÿÿ2
p

 
ª "ÿÿÿÿÿÿÿÿÿ2
,__inference_autoencoder_layer_call_fn_862712i#$%&'()*+,-.;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ2
p 

 
ª "ÿÿÿÿÿÿÿÿÿ2
,__inference_autoencoder_layer_call_fn_862741i#$%&'()*+,-.;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ2
p

 
ª "ÿÿÿÿÿÿÿÿÿ2­
E__inference_conv1d_22_layer_call_and_return_conditional_losses_863161d#$3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ2
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ.
 
*__inference_conv1d_22_layer_call_fn_863145W#$3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ2
ª "ÿÿÿÿÿÿÿÿÿ.­
E__inference_conv1d_23_layer_call_and_return_conditional_losses_863186d%&3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ.
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ*
 
*__inference_conv1d_23_layer_call_fn_863170W%&3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ.
ª "ÿÿÿÿÿÿÿÿÿ*º
C__inference_decoder_layer_call_and_return_conditional_losses_862388s)*+,-.>¢;
4¢1
'$
decoder_inputÿÿÿÿÿÿÿÿÿ@
p 

 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ2
 º
C__inference_decoder_layer_call_and_return_conditional_losses_862408s)*+,-.>¢;
4¢1
'$
decoder_inputÿÿÿÿÿÿÿÿÿ@
p

 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ2
 ³
C__inference_decoder_layer_call_and_return_conditional_losses_863084l)*+,-.7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ@
p 

 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ2
 ³
C__inference_decoder_layer_call_and_return_conditional_losses_863117l)*+,-.7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ@
p

 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ2
 
(__inference_decoder_layer_call_fn_862261f)*+,-.>¢;
4¢1
'$
decoder_inputÿÿÿÿÿÿÿÿÿ@
p 

 
ª "ÿÿÿÿÿÿÿÿÿ2
(__inference_decoder_layer_call_fn_862368f)*+,-.>¢;
4¢1
'$
decoder_inputÿÿÿÿÿÿÿÿÿ@
p

 
ª "ÿÿÿÿÿÿÿÿÿ2
(__inference_decoder_layer_call_fn_863034_)*+,-.7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ@
p 

 
ª "ÿÿÿÿÿÿÿÿÿ2
(__inference_decoder_layer_call_fn_863051_)*+,-.7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ@
p

 
ª "ÿÿÿÿÿÿÿÿÿ2¤
D__inference_dense_33_layer_call_and_return_conditional_losses_863236\)*/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ 
 |
)__inference_dense_33_layer_call_fn_863225O)*/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "ÿÿÿÿÿÿÿÿÿ ¤
D__inference_dense_34_layer_call_and_return_conditional_losses_863256\+,/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ 
 |
)__inference_dense_34_layer_call_fn_863245O+,/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "ÿÿÿÿÿÿÿÿÿ ¤
D__inference_dense_35_layer_call_and_return_conditional_losses_863275\-./¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ2
 |
)__inference_dense_35_layer_call_fn_863265O-./¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "ÿÿÿÿÿÿÿÿÿ2º
C__inference_encoder_layer_call_and_return_conditional_losses_862152s#$%&'(B¢?
8¢5
+(
encoder_inputÿÿÿÿÿÿÿÿÿ2
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 º
C__inference_encoder_layer_call_and_return_conditional_losses_862173s#$%&'(B¢?
8¢5
+(
encoder_inputÿÿÿÿÿÿÿÿÿ2
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 ³
C__inference_encoder_layer_call_and_return_conditional_losses_862971l#$%&'(;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ2
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 ³
C__inference_encoder_layer_call_and_return_conditional_losses_863017l#$%&'(;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ2
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 
(__inference_encoder_layer_call_fn_862017f#$%&'(B¢?
8¢5
+(
encoder_inputÿÿÿÿÿÿÿÿÿ2
p 

 
ª "ÿÿÿÿÿÿÿÿÿ@
(__inference_encoder_layer_call_fn_862131f#$%&'(B¢?
8¢5
+(
encoder_inputÿÿÿÿÿÿÿÿÿ2
p

 
ª "ÿÿÿÿÿÿÿÿÿ@
(__inference_encoder_layer_call_fn_862908_#$%&'(;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ2
p 

 
ª "ÿÿÿÿÿÿÿÿÿ@
(__inference_encoder_layer_call_fn_862925_#$%&'(;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ2
p

 
ª "ÿÿÿÿÿÿÿÿÿ@§
F__inference_flatten_11_layer_call_and_return_conditional_losses_863197]3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ*
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ¨
 
+__inference_flatten_11_layer_call_fn_863191P3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ*
ª "ÿÿÿÿÿÿÿÿÿ¨ª
I__inference_latent_vector_layer_call_and_return_conditional_losses_863216]'(0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ¨
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 
.__inference_latent_vector_layer_call_fn_863206P'(0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ¨
ª "ÿÿÿÿÿÿÿÿÿ@®
J__inference_random_noise_4_layer_call_and_return_conditional_losses_863136`3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ2
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ2
 
/__inference_random_noise_4_layer_call_fn_863122S3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ2
ª "ÿÿÿÿÿÿÿÿÿ2¦
F__inference_reshape_11_layer_call_and_return_conditional_losses_863293\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ2
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ2
 ~
+__inference_reshape_11_layer_call_fn_863280O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ2
ª "ÿÿÿÿÿÿÿÿÿ2»
$__inference_signature_wrapper_862683#$%&'()*+,-.K¢H
¢ 
Aª>
<
encoder_input+(
encoder_inputÿÿÿÿÿÿÿÿÿ2"5ª2
0
decoder%"
decoderÿÿÿÿÿÿÿÿÿ2