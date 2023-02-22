/*
Submit problemtype to Flask in step3 limited condition
*/

//Ajax通信でデータ受け渡し
$(function () {

    //入力フォーム、プルダウンから値を抽出
    function extractVals(div) {
        var vals = [];
        div.each(function () {
            vals.push($(this).val());
        });

        return vals;
    };

    //チェックボックスから値を抽出
    function checkboxVals(div) {
        var vals = [];
        div.each(function () {
            if ($(this).prop('checked')) {
                vals.push($(this).val());
            }
            else {
                vals.push('');
            }
        });

        return vals;
    };

    //submit押下時にFlask側へデータを送信
    $('#btn_submit_condition').click(function (event) {
        event.preventDefault();

        //処理中はボタン非活性
        btn_disable();

        //sampling
        let sampling_num = $('input[name="sampling_num"]').val(); 
        let sampling_width = $('input[name="sampling_width"]').val()
        let nega_flag = document.getElementById("nega_flag").checked

        //range
        var range_sel = $('select[name="range_sel[]"]');
        var range_sel_vals = extractVals(range_sel);
        var range_lower = $('input[name="range_lower[]"]');
        var range_lower_vals = extractVals(range_lower);
        var range_upper = $('input[name="range_upper[]"]');
        var range_upper_vals = extractVals(range_upper);

        //fix
        var fixed_sel = $('select[name="fixed_sel[]"]');
        var fixed_sel_vals = extractVals(fixed_sel);
        var fixed = $('input[name="fixed_value[]"]');
        var fixed_vals = extractVals(fixed);

        //total
        var total_cb = $('input[name="total_cb[]"]');
        var total_cb_vals = checkboxVals(total_cb);
        var total = $('input[name="total_value[]"]');
        var total_vals = extractVals(total);

        //combination
        var combination_cb = $('input[name="combination_cb[]"]');
        var combination_cb_vals = checkboxVals(combination_cb);
        var combination_lower = $('input[name="combination_lower[]"]');
        var combination_lower_vals = extractVals(combination_lower);
        var combination_upper = $('input[name="combination_upper[]"]');
        var combination_upper_vals = extractVals(combination_upper);

        //ratio
        var ratio1_sel = $('select[name="ratio1_sel[]"]');
        var ratio1_sel_vals = extractVals(ratio1_sel);
        var ratio2_sel = $('select[name="ratio2_sel[]"]');
        var ratio2_sel_vals = extractVals(ratio2_sel);
        var ratio1 = $('input[name="ratio1_value[]"]');
        var ratio1_vals = extractVals(ratio1);
        var ratio2 = $('input[name="ratio2_value[]"]');
        var ratio2_vals = extractVals(ratio2);

        //groupsum
        let group_cb_list = new Array(5);
        let group_cb_vals_list = new Array(5);
        let group_lower_list = new Array(5);
        let group_upper_list = new Array(5);
        let group_lower_vals_list = new Array(5);
        let group_upper_vals_list = new Array(5);
        for (let i = 0; i < 5; i++) {
            group_cb_list[i] = $('#id_groupsum_' + i + ' input[name="group_cb[]"]');
            group_cb_vals_list[i] = checkboxVals(group_cb_list[i]);
            group_lower_list[i] = $('#id_groupsum_' + i + ' input[name="group_value_lower[]"]');
            group_upper_list[i] = $('#id_groupsum_' + i + ' input[name="group_value_upper[]"]');
            group_lower_vals_list[i] = extractVals(group_lower_list[i]);
            group_upper_vals_list[i] = extractVals(group_upper_list[i]);
        }
        let group_total = $('input[name="group_total[]"]');
        let group_total_vals = extractVals(group_total);

        //json形式へ変換
        var json_data = {
            "sampling_num": sampling_num, "sampling_width": sampling_width, "nega_flag": nega_flag,
            "range_sel": range_sel_vals, "range_lower": range_lower_vals, "range_upper": range_upper_vals,
            "fixed_sel": fixed_sel_vals, "fixed_val": fixed_vals,
            "total_cb": total_cb_vals, "total_val": total_vals,
            "combination_cb": combination_cb_vals, "combination_lower": combination_lower_vals, "combination_upper": combination_upper_vals,
            "ratio1_sel": ratio1_sel_vals, "ratio2_sel": ratio2_sel_vals, "ratio1_val": ratio1_vals, "ratio2_val": ratio2_vals,
            "group_cb_list": group_cb_vals_list, "group_lower_list": group_lower_vals_list, "group_upper_list": group_upper_vals_list, "group_total": group_total_vals
        };

        //Ajax通信
        $.ajax({
            data: JSON.stringify(json_data),
            type: 'POST',
            url: '/create_sample/' + exp_id,
            dataType: 'json',
            contentType: 'application/json'
        }).done(function (data) {
            console.log(data);
        }).fail(function () {
            location.reload();
        });
        
        return false;
    });
});