class EncoderDecoderConvLSTM(nn.Module):
    def __init__(self, nf, in_chan):
        super(EncoderDecoderConvLSTM, self).__init__()

        """ 구조

        # Encoder (ConvLSTM)
        # Encoder Vector - context vector
        # Decoder (ConvLSTM) - Encoder Vector를 input으로 적용합니다.
        # Decoder (3D CNN) - 모형에 대한 회귀 예측을 생성합니다.

        """
        
        """
        
        # ConvLSTMCell은 h_next,c_next를 반환합니다.
        (self, input_dim, hidden_dim, kernel_size, bias)
        
        """
        self.encoder_1_convlstm = ConvLSTMCell(input_dim=in_chan,
                                               hidden_dim=nf,
                                               kernel_size=(3, 3),
                                               bias=True)

        self.encoder_2_convlstm = ConvLSTMCell(input_dim=nf,
                                               hidden_dim=nf,
                                               kernel_size=(3, 3),
                                               bias=True)

        self.encoder_3_convlstm = ConvLSTMCell(input_dim=nf,
                                               hidden_dim=nf,
                                               kernel_size=(3, 3),
                                               bias=True)

        self.encoder_4_convlstm = ConvLSTMCell(input_dim=nf,
                                               hidden_dim=nf,
                                               kernel_size=(3, 3),
                                               bias=True)

        self.decoder_1_convlstm = ConvLSTMCell(input_dim=nf, 
                                               hidden_dim=nf,
                                               kernel_size=(3, 3),
                                               bias=True)

        self.decoder_2_convlstm = ConvLSTMCell(input_dim=nf,
                                               hidden_dim=nf,
                                               kernel_size=(3, 3),
                                               bias=True)

        self.decoder_3_convlstm = ConvLSTMCell(input_dim=nf,
                                               hidden_dim=nf,
                                               kernel_size=(3, 3),
                                               bias=True)

        self.decoder_4_convlstm = ConvLSTMCell(input_dim=nf,
                                               hidden_dim=nf,
                                               kernel_size=(3, 3),
                                               bias=True)
        
        """
        
        in_channels로 받은 nf개의 채널을 입력받아 1개의 채널로 출력하고,
        커널은 첫번째 차원으 1로 유지, 두세번째 차원에서 3*3 필터를 사용해 합성곱을 수행합니다.
        padding으로 두번째 세번째 차원에 각각 1씩의 패딩을 추가합니다.
        
        """
        self.decoder_CNN = nn.Conv3d(in_channels=nf,
                                     out_channels=1,
                                     kernel_size=(1, 3, 3),
                                     padding=(0, 1, 1))


    def autoencoder(self, x, seq_len, future_step, h_t, c_t, h_t2, c_t2, h_t3, c_t3, h_t4, c_t4, h_t5, c_t5, h_t6, c_t6, h_t7, c_t7, h_t8, c_t8):

        outputs = []

        # encoder
        for t in range(seq_len):  
            # x[samples, timesteps, rows, columns, features]
            h_t, c_t = self.encoder_1_convlstm(input_tensor=x[:, t, :, :, :],   
                                               cur_state=[h_t, c_t])      # concatenate를 이용해 두 개의 텐서를 연결하여 skip connection을 제공할 수 있습니다.
            h_t2, c_t2 = self.encoder_2_convlstm(input_tensor=h_t,
                                                 cur_state=[h_t2, c_t2])  # concatenate를 이용해 두 개의 텐서를 연결하여 skip connection을 제공할 수 있습니다.
            h_t3, c_t3 = self.encoder_3_convlstm(input_tensor=h_t2,
                                                 cur_state=[h_t3, c_t3])  # concatenate를 이용해 두 개의 텐서를 연결하여 skip connection을 제공할 수 있습니다.
            h_t4, c_t4 = self.encoder_4_convlstm(input_tensor=h_t3,
                                                 cur_state=[h_t4, c_t4])  # concatenate를 이용해 두 개의 텐서를 연결하여 skip connection을 제공할 수 있습니다.
        # encoder_vector
        # decoder의 input
        encoder_vector = h_t4 

        # decoder
        for t in range(future_step):
            h_t5, c_t5 = self.decoder_1_convlstm(input_tensor=encoder_vector,
                                                 cur_state=[h_t5, c_t5])  # concatenate를 이용해 두 개의 텐서를 연결하여 skip connection을 제공할 수 있습니다.
            h_t6, c_t6 = self.decoder_2_convlstm(input_tensor=h_t5,
                                                 cur_state=[h_t6, c_t6])  # concatenate를 이용해 두 개의 텐서를 연결하여 skip connection을 제공할 수 있습니다.
            h_t7, c_t7 = self.decoder_3_convlstm(input_tensor=h_t6,
                                                 cur_state=[h_t7, c_t7])  # concatenate를 이용해 두 개의 텐서를 연결하여 skip connection을 제공할 수 있습니다.
            h_t8, c_t8 = self.decoder_4_convlstm(input_tensor=h_t7,
                                                 cur_state=[h_t8, c_t8])  # concatenate를 이용해 두 개의 텐서를 연결하여 skip connection을 제공할 수 있습니다.
            encoder_vector = h_t8
            outputs += [h_t8]  # predictions


        outputs = torch.stack(outputs, 1)    # 1번째 차원을 따라서 텐서를 생성합니다.
        outputs = outputs.permute(0, 2, 1, 3, 4) # 두번째 차원과 세번째 차원의 순서를 바꿉니다.
        outputs = self.decoder_CNN(outputs)
        outputs = torch.nn.Sigmoid()(outputs)

        return outputs
    
    

    def forward(self, x, future_seq=1, hidden_state=None):

        """
        Parameters
        ----------
        input_tensor:
            5-D Tensor of shape (b, t, c, h, w)        #   batch, time, channel, height, width
        """

        # 서로 다른 입력 차원 수의 size를 찾습니다.
        b, seq_len, _, h, w = x.size()

        # hidden state를 초기화합니다.
        h_t, c_t = self.encoder_1_convlstm.init_hidden(batch_size=b, image_size=(h, w))
        h_t2, c_t2 = self.encoder_2_convlstm.init_hidden(batch_size=b, image_size=(h, w))
        h_t3, c_t3 = self.encoder_3_convlstm.init_hidden(batch_size=b, image_size=(h, w))
        h_t4, c_t4 = self.encoder_4_convlstm.init_hidden(batch_size=b, image_size=(h, w))
        h_t5, c_t5 = self.decoder_1_convlstm.init_hidden(batch_size=b, image_size=(h, w))
        h_t6, c_t6 = self.decoder_2_convlstm.init_hidden(batch_size=b, image_size=(h, w))
        h_t7, c_t7 = self.decoder_3_convlstm.init_hidden(batch_size=b, image_size=(h, w))
        h_t8, c_t8 = self.decoder_4_convlstm.init_hidden(batch_size=b, image_size=(h, w))

        # autoencoder forward
        outputs = self.autoencoder(x, seq_len, future_seq, h_t, c_t, h_t2, c_t2, h_t3, c_t3, h_t4, c_t4, h_t5, c_t5, h_t6, c_t6, h_t7, c_t7, h_t8, c_t8)

        return outputs
