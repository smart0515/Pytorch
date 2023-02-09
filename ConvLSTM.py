import torch
import torch.nn as nn

class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        # 입력 텐서는 input_channels + hidden_channels 개의 채널을 가지고 있고, 출력 텐서는 4 * hidden_channels 개의 채널을 가집니다. 
        # 합성곱의 커널 크기는 kernel_size로 설정되어 있으며, 패딩은 (kernel_size - 1) // 2로 설정되어 있어서 출력의 공간 차원이 입력과 동일하도록 합니다.
        self.conv = nn.Conv2d(input_channels + hidden_channels, 4 * hidden_channels, kernel_size, padding=(kernel_size - 1) // 2)

    def forward(self, input, state):
        h, c = state
    
        # 입력 텐서와 hidden state 텐서를 채널 차원으로 합칩니다.
        combined = torch.cat((input, h), dim=1)  
        
        # 합쳐진 텐서에 convolution을 적용합니다.
        combined_conv = self.conv(combined)     
        
        # 합쳐진 convolution 결과를 4개의 부분으로 분리합니다.
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_channels, dim=1)  

        i = torch.sigmoid(cc_i)      # input gate에 sigmoid 활성화 함수를 적용합니다.
        f = torch.sigmoid(cc_f)      # forget gate에 sigmoid 활성화 함수를 적용합니다.
        o = torch.sigmoid(cc_o)      # output gate에 sigmoid 활성화 함수를 적용합니다.
        g = torch.tanh(cc_g)         # cell을 나타내는 값에 tanh 활성화 함수를 적용합니다.
        c_next = f * c + i * g       # 다음 cell state를 계산합니다.
        h_next = o * torch.tanh(c_next)  # 다음 hidden state를 계산합니다.
        return h_next, (h_next, c_next)
    
class ConvLSTM(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, num_layers):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        # list comprehension을 사용하여 ConvLSTMCell 생성합니다.
        self.lstm_cells = nn.ModuleList([ConvLSTMCell(input_channels if i == 0 else hidden_channels, hidden_channels, kernel_size) for i in range(num_layers)])

    def forward(self, input):
        # 텐서 input을 받아서 ConvLSTM의 마지막 hidden state를 반환합니다.
        # ConvLSTM의 상태를 입력 텐서와 동일한 모양의 두개의 0 텐서로 초기화합니다.
        # 이 0 텐서는 첫 번째 ConvLSTMCell에서의 초기 hidden과 cell state로 사용됩니다.
        state = (torch.zeros(input.size(0), self.hidden_channels, input.size(2), input.size(3)).to(input.device),
                 torch.zeros(input.size(0), self.hidden_channels, input.size(2), input.size(3)).to(input.device))
        for i, lstm_cell in enumerate(self.lstm_cells):
            input, state = lstm_cell(input, state)
        return state[0]
